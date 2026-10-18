#!/usr/bin/env python3
"""The generated coefficient set's census, shared by the two checkers that compile against it.

`tools/gen_boys_coefficients.py` emits one master and one header per coefficient group, and a
translation unit that names a policy outside the default set must open the group header its fit
names. The set is therefore a table of the library like any other, and this module reads it back
from a revision's text: every group tag in the master has one header, that header defines an array
and marks the tag present, every marker read in the hand-written headers names a declared tag, the
front door's default set is a subset of the group headers, and the umbrella opens all of them.

The callers supply the text they read, from a git revision or from an extracted tree, so this
module states the rule once and neither checker re-states it.
"""

from __future__ import annotations

import re

MASTER = "include/boys/boys_coefficients.hpp"
# The spelling a header reaches the master by: an include names `boys/...`, and a repository
# path carries the `include/` in front of it.
MASTER_INCLUDE = MASTER[len("include/"):]
DEFAULT_TABLES = "include/boys/boys_default_tables.hpp"
UMBRELLA = "include/boys/boys_all.hpp"

# Every file the generator's artifact-set rule owns: the master and one header per group.
GROUP_HEADER = re.compile(r"^include/boys/boys_coefficients_[A-Za-z0-9_]+\.hpp$")
TAG = re.compile(r"^struct\s+([A-Za-z0-9]+Group)\s*\{\s*\}\s*;\s*$", re.M)
SPECIALIZATION = re.compile(
    r"^template\s*<>\s*inline constexpr bool\s+kCoefficientGroupPresent<\s*"
    r"([A-Za-z0-9]+)\s*>\s*=\s*true\s*;\s*$", re.M)
PRIMARY = "template <class Group> inline constexpr bool kCoefficientGroupPresent = false;"
ARRAY = "inline constexpr std::array<"
INCLUDE = re.compile(r'^\s*#\s*include\s+[<"](boys/[A-Za-z0-9_./]+)[>"]', re.M)


def _matching(text: str, start: int, opening: str = "<", closing: str = ">") -> int:
    """The index just past the bracket matching the one at start."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def marker_reads(text: str) -> tuple[list[str], int]:
    """The group tags a text reads through the marker, and how many reads it holds.

    A read is either `kCoefficientGroupPresent<Tag>` or the dependent spelling the fits use,
    `kCoefficientGroupPresent<DependentGroup<Carrier, Tag>>`: the tag is the last identifier of
    the marker's argument in both.
    """
    tags = []
    count = 0
    position = 0
    needle = "kCoefficientGroupPresent<"
    while True:
        at = text.find(needle, position)
        if at < 0:
            return tags, count
        end = _matching(text, at + len(needle) - 1)
        if end < 0:
            return tags, count
        argument = text[at + len(needle):end - 1].strip()
        count += 1
        position = end
        # The tag is the identifier immediately before the argument's closing bracket. The
        # dependent spelling carries it after the last top-level comma; the plain spelling is
        # the identifier itself.
        if argument.startswith("DependentGroup<"):
            inner = argument[len("DependentGroup<"):-1]
            depth = 0
            comma = -1
            for index, character in enumerate(inner):
                if character == "<":
                    depth += 1
                elif character == ">":
                    depth -= 1
                elif character == "," and depth == 0:
                    comma = index
            if comma < 0:
                tags.append("")
                continue
            tags.append(inner[comma + 1:].strip())
        else:
            tags.append(argument)


def census(master: str, groups: dict[str, str], scanners: dict[str, str],
           default_tables: str, umbrella: str) -> tuple[dict, list[str]]:
    """The set's census: the counts it read and every rule it found broken."""
    findings = []

    declared = TAG.findall(master)
    if PRIMARY not in master:
        findings.append(f"{MASTER} states no marker primary ({PRIMARY!r})")
    if not declared:
        findings.append(f"{MASTER} declares no group tag; the set's census read nothing")

    header_tags: dict[str, str] = {}
    for path, text in sorted(groups.items()):
        specializations = SPECIALIZATION.findall(text)
        if len(specializations) != 1:
            findings.append(f"{path} marks {len(specializations)} group(s) present; a group "
                            f"header marks exactly its own tag")
            continue
        tag = specializations[0]
        header_tags[path] = tag
        if tag not in declared:
            findings.append(f"{path} marks {tag} present, and {MASTER} declares no such tag")
        if ARRAY not in text:
            findings.append(f"{path} states no stored array: it is a group header and the "
                            f"generator emits one per family")
        if MASTER_INCLUDE not in INCLUDE.findall(text):
            findings.append(f"{path} does not include {MASTER}: a fit parses against the master "
                            f"and reads the group's arrays from this header")
        if text.count("template <>") > 1:
            findings.append(f"{path} holds more than one specialization: the artifact-set rule "
                            f"is one marker per group header")

    missing = sorted(set(declared) - set(header_tags.values()))
    duplicated = sorted({tag for tag in header_tags.values()
                         if list(header_tags.values()).count(tag) > 1})
    if missing:
        findings.append(f"{MASTER} declares {len(missing)} tag(s) no group header marks present: "
                        + ", ".join(missing))
    if duplicated:
        findings.append("more than one group header marks the same tag present: "
                        + ", ".join(duplicated))

    reads = 0
    for path, text in sorted(scanners.items()):
        tags, count = marker_reads(text)
        reads += count
        unknown = sorted({tag for tag in tags if tag not in declared})
        if unknown:
            findings.append(f"{path} reads the marker for {', '.join(unknown)}: the tag is not "
                            f"one {MASTER} declares")

    default_headers = [path for path in INCLUDE.findall(default_tables) if path.startswith(
        "boys/boys_coefficients")]
    if not default_headers:
        findings.append(f"{DEFAULT_TABLES} opens no group header: the front door would compile "
                        f"against the master alone")
    for relative in default_headers:
        path = "include/" + relative
        if path not in groups:
            findings.append(f"{DEFAULT_TABLES} opens {relative}, which is not a group header the "
                            f"generator emits")

    umbrella_headers = set(INCLUDE.findall(umbrella))
    for path in sorted(groups):
        relative = path[len("include/"):]
        if relative not in umbrella_headers:
            findings.append(f"{UMBRELLA} does not open {relative}: the umbrella opens every "
                            f"group, which is what makes it a consumer's one include")

    summary = {"declared": len(declared), "group_headers": len(groups),
               "marker_reads": reads, "default_groups": len(default_headers)}
    return summary, findings
