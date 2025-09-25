"""Griffe Fieldz extension."""

from __future__ import annotations

import inspect
import textwrap
import types
import typing
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterable, Literal, TypedDict, TypeVar, cast

import fieldz
import typing_extensions
from fieldz._repr import typing_base, WithArgsTypes, origin_is_literal, origin_is_union
from griffe import (
    Attribute,
    Class,
    Docstring,
    DocstringAttribute,
    DocstringParameter,
    DocstringSection,
    DocstringSectionAttributes,
    DocstringSectionParameters,
    Extension,
    Object,
    ObjectNode,
    dynamic_import,
    get_logger,
    parse_docstring_annotation,
)

if TYPE_CHECKING:
    import ast

    from griffe import Expr, Inspector, Visitor

    AddFieldsTo = Literal[
        "docstring-parameters", "docstring-attributes", "class-attributes"
    ]

logger = get_logger("griffe-fieldz")


class FieldzExtension(Extension):
    """Griffe extension that injects field information for dataclass-likes."""

    def __init__(
            self,
            object_paths: list[str] | None = None,
            include_private: bool = False,
            include_inherited: bool = False,
            add_fields_to: AddFieldsTo = "docstring-parameters",
            remove_fields_from_members: bool = False,
            **kwargs: Any,
    ) -> None:
        self.object_paths = object_paths
        self._kwargs = kwargs
        if kwargs:
            logger.warning(
                "Unknown kwargs passed to FieldzExtension: %s", ", ".join(kwargs)
            )
        self.include_private = include_private
        self.include_inherited = include_inherited

        self.remove_fields_from_members = remove_fields_from_members
        if add_fields_to not in (
                "docstring-parameters",
                "docstring-attributes",
                "class-attributes",
        ):  # pragma: no cover
            logger.error(
                "'add_fields_to' must be one of {'docstring-parameters', "
                f"'docstring-attributes', or 'class-attributes'}}, not {add_fields_to}."
                "\n\nDefaulting to 'docstring-parameters'."
            )
            add_fields_to = "docstring-parameters"

        self.add_fields_to: AddFieldsTo = add_fields_to

    def on_class_members(
            self,
            *,
            node: ast.AST | ObjectNode,
            cls: Class,
            agent: Visitor | Inspector,
            **kwargs: Any,
    ) -> None:
        if isinstance(node, ObjectNode):
            return  # skip runtime objects

        if self.object_paths and cls.path not in self.object_paths:
            return  # skip objects that were not selected

        # import object to get its evaluated docstring
        try:
            runtime_obj = dynamic_import(cls.path)
        except ImportError:
            logger.debug(f"Could not get dynamic docstring for {cls.path}")
            return

        try:
            fieldz.get_adapter(runtime_obj)
        except TypeError:  # pragma: no cover
            return
        self._inject_fields(cls, runtime_obj)

    # ------------------------------

    def _inject_fields(self, griffe_obj: Object, runtime_obj: Any) -> None:
        # update the object instance with the evaluated docstring
        docstring = inspect.cleandoc(getattr(runtime_obj, "__doc__", "") or "")
        if not griffe_obj.docstring:
            griffe_obj.docstring = Docstring(docstring, parent=griffe_obj)

        # collect field info
        fields = fieldz.fields(runtime_obj)
        if not self.include_inherited:
            annotations = getattr(runtime_obj, "__annotations__", {})
            fields = tuple(f for f in fields if f.name in annotations)

        _unify_fields(
            fields,
            griffe_obj,
            include_private=self.include_private,
            add_fields_to=self.add_fields_to,
            remove_fields_from_members=self.remove_fields_from_members,
        )


def _display_as_type(obj: Any, *, modern_union: bool = False) -> str:
    """Pretty representation of a type.

    Should be as close as possible to the original type definition string.
    Takes some logic from `typing._type_repr`.
    """
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    elif obj is ...:
        return "..."
    elif obj in (None, type(None)):
        return "None"

    if not isinstance(obj, (typing_base, WithArgsTypes, type, typing.TypeAliasType, typing.TypeVar, typing.NewType)):
        obj = obj.__class__

    if isinstance(obj, (typing.NewType, typing.TypeVar)):
        # TypeVar repr includes a prepended ~ and NewType repr includes the module name prepended,
        # so we use __name__ to get a clean name
        return obj.__name__

    origin = typing_extensions.get_origin(obj)
    if origin_is_literal(origin):
        # For Literal types, represent the actual values, not their types
        arg_reprs = [repr(arg) for arg in typing_extensions.get_args(obj)]
        return f"Literal[{', '.join(arg_reprs)}]"
    elif origin_is_union(origin):
        args = [_display_as_type(x) for x in typing_extensions.get_args(obj)]
        if modern_union:
            return " | ".join(args)
        if len(args) == 2 and "None" in args:
            args.remove("None")
            return f"Optional[{args[0]}]"
        return f"Union[{', '.join(args)}]"
    elif isinstance(obj, WithArgsTypes):
        argstr = ", ".join(map(_display_as_type, typing_extensions.get_args(obj)))
        return f"{obj.__qualname__}[{argstr}]"
    elif isinstance(obj, type):
        return obj.__qualname__
    else:  # pragma: no cover
        return repr(obj).replace("typing.", "").replace("typing_extensions.", "")


def _to_annotation(type_: Any, docstring: Docstring) -> str | Expr | None:
    """Create griffe annotation for a type."""
    if type_:
        return parse_docstring_annotation(
            _display_as_type(type_, modern_union=True), docstring
        )
    return None


def _default_repr(field: fieldz.Field) -> str | None:
    """Return a repr for a field default."""
    try:
        if field.default is not field.MISSING:
            return repr(field.default)
        if (factory := field.default_factory) is not field.MISSING:
            try:
                sig = inspect.signature(factory)
            except ValueError:
                return repr(factory)
            else:
                if len(sig.parameters) == 0:
                    with suppress(Exception):
                        return repr(factory())  # type: ignore[call-arg]

            return "<dynamic>"
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to get default repr for %s: %s", field.name, exc)
        pass
    return None


class DocstringNamedElementKwargs(TypedDict):
    """Docstring named element kwargs."""

    name: str
    description: str
    annotation: str | Expr | None
    value: str | None


def _unify_fields(
        fields: Iterable[fieldz.Field],
        griffe_obj: Object,
        include_private: bool,
        add_fields_to: AddFieldsTo,
        remove_fields_from_members: bool,
) -> None:
    docstring = cast("Docstring", griffe_obj.docstring)
    sections = docstring.parsed

    for field in fields:
        if not include_private and field.name.startswith("_"):
            continue

        try:
            item_kwargs = _merged_kwargs(field, docstring, griffe_obj)

            if add_fields_to == "class-attributes":
                if field.name not in griffe_obj.attributes:
                    griffe_obj.members[field.name] = Attribute(
                        name=item_kwargs["name"],
                        value=item_kwargs["value"],
                        annotation=item_kwargs["annotation"],
                        docstring=item_kwargs["description"],
                    )
            elif add_fields_to == "docstring-attributes" or (not field.init):
                _add_if_missing(sections, DocstringAttribute(**item_kwargs))
                # remove from parameters if it exists
                if p_sect := _get_section(sections, DocstringSectionParameters):
                    p_sect.value = [x for x in p_sect.value if x.name != field.name]
                if remove_fields_from_members:
                    # remove from griffe_obj.parameters
                    griffe_obj.members.pop(field.name, None)
                    griffe_obj.inherited_members.pop(field.name, None)
            elif add_fields_to == "docstring-parameters":
                _add_if_missing(sections, DocstringParameter(**item_kwargs))
                # remove from attributes if it exists
                if a_sect := _get_section(sections, DocstringSectionAttributes):
                    a_sect.value = [x for x in a_sect.value if x.name != field.name]
                if remove_fields_from_members:
                    # remove from griffe_obj.attributes
                    griffe_obj.members.pop(field.name, None)
                    griffe_obj.inherited_members.pop(field.name, None)

        except Exception as exc:
            logger.warning("Failed to parse field %s: %s", field.name, exc)


def _merged_kwargs(
        field: fieldz.Field, docstring: Docstring, griffe_obj: Object
) -> DocstringNamedElementKwargs:
    desc = field.description or field.metadata.get("description", "") or ""
    if not desc and (doc := getattr(field.default_factory, "__doc__", None)):
        desc = inspect.cleandoc(doc) or ""

    if not desc and field.name in griffe_obj.attributes:
        griffe_attr = griffe_obj.attributes[field.name]
        if griffe_attr.docstring:
            desc = griffe_attr.docstring.value

    return DocstringNamedElementKwargs(
        name=field.name,
        description=textwrap.dedent(desc).strip(),
        annotation=_to_annotation(field.type, docstring),
        value=_default_repr(field),
    )


T = TypeVar("T", bound="DocstringSectionParameters | DocstringSectionAttributes")


def _get_section(sections: list[DocstringSection], cls: type[T]) -> T | None:
    for section in sections:
        if isinstance(section, cls):
            return section
    return None


def _add_if_missing(
        sections: list[DocstringSection], item: DocstringParameter | DocstringAttribute
) -> None:
    section: DocstringSectionParameters | DocstringSectionAttributes | None
    if isinstance(item, DocstringParameter):
        if not (section := _get_section(sections, DocstringSectionParameters)):
            section = DocstringSectionParameters([])
            sections.append(section)
    elif isinstance(item, DocstringAttribute):
        if not (section := _get_section(sections, DocstringSectionAttributes)):
            section = DocstringSectionAttributes([])
            sections.append(section)
    else:  # pragma: no cover
        raise TypeError(f"Unknown section type: {type(item)}")

    existing = {x.name: x for x in section.value}
    if item.name in existing:
        current = existing[item.name]
        if current.description is None and item.description:
            current.description = item.description
        if current.annotation is None and item.annotation:
            current.annotation = item.annotation
        if current.value is None and item.value is not None:
            current.value = item.value
    else:
        section.value.append(item)  # type: ignore [arg-type]
