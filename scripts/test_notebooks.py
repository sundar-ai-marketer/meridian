# Copyright 2026 Sundar Ramesh Kumar.
# Licensed under the Apache License, Version 2.0.
# NOTICE: This file is new in this fork; see NOTICE at the repository root.

"""Static contracts for the shipped demo notebooks.

The notebooks cannot be executed here, and saying why is more useful than a
skipped test: five of the nine import `google.colab` (`drive.mount`,
`userdata`), which does not exist off Colab; all nine carry `pip install`
cells that would rewrite the environment they run in; and they sample at
production settings -- ten chains, two thousand adaptation draws -- which is
hours per notebook. Executing them belongs on Colab, against a real dataset.

What can be checked without running them is whether they would work at all,
and that is what this does: the JSON parses, every `meridian` import resolves
against the installed package, every repository file they reference exists,
and none of them teaches an API this fork has moved away from.

The last one is the reason this file exists. `Meridian_RF_Demo.ipynb` told
readers to save a fitted model with `model.save_mmm`, the deprecated
pickle path, in a step headed "We recommend that you save the model object" --
while `skills/meridian_model_building/SKILL.md` tells an agent not to use it.
A tutorial is documentation, and it drifts the same way prose does.
"""

import json
import pathlib
import re
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_NOTEBOOKS = sorted((_REPO_ROOT / "demo").glob("*.ipynb"))

# `from meridian.x import y` / `import meridian.x`
_MERIDIAN_IMPORT = re.compile(
    r"^\s*(?:from\s+(meridian[\w.]*)\s+import\s|import\s+(meridian[\w.]*))",
    re.MULTILINE,
)
# A quoted path inside the repository, e.g. 'meridian/data/.../geo_media.csv'.
_REPO_PATH = re.compile(
    r"['\"]((?:meridian|demo|docs|examples|scripts)/[\w./-]+"
    r"\.(?:csv|pkl|xlsx|nc|json|binpb|scss|css))['\"]"
)
_SAVE_MMM = re.compile(r"\bmodel\.save_mmm\s*\(")
_LOAD_MMM = re.compile(r"\bmodel\.load_mmm\s*\(")


def _code(notebook: pathlib.Path) -> str:
  document = json.loads(notebook.read_text(encoding="utf-8"))
  return "\n".join(
      "".join(cell.get("source", []))
      for cell in document.get("cells", [])
      if cell.get("cell_type") == "code"
  )


class NotebookStructureTest(unittest.TestCase):

  def test_there_are_notebooks_to_check(self):
    # A glob that silently matches nothing would make every test below pass.
    self.assertTrue(_NOTEBOOKS, f"no notebooks found under {_REPO_ROOT/'demo'}")

  def test_every_notebook_is_valid_json_with_cells(self):
    for notebook in _NOTEBOOKS:
      with self.subTest(notebook=notebook.name):
        document = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertIn("cells", document)
        self.assertTrue(document["cells"], "notebook has no cells")
        for cell in document["cells"]:
          self.assertIn(cell.get("cell_type"), ("code", "markdown", "raw"))
          self.assertIsInstance(cell.get("source"), list)


class NotebookImportTest(unittest.TestCase):

  def test_every_meridian_import_resolves(self):
    for notebook in _NOTEBOOKS:
      code = _code(notebook)
      modules = {
          match.group(1) or match.group(2)
          for match in _MERIDIAN_IMPORT.finditer(code)
      }
      for module in sorted(modules):
        with self.subTest(notebook=notebook.name, module=module):
          try:
            __import__(module)
          except ImportError as error:
            self.fail(f"{notebook.name} imports {module}, which fails: {error}")


class NotebookReferenceTest(unittest.TestCase):

  def test_every_referenced_repository_file_exists(self):
    for notebook in _NOTEBOOKS:
      for reference in sorted(set(_REPO_PATH.findall(_code(notebook)))):
        with self.subTest(notebook=notebook.name, path=reference):
          self.assertTrue(
              (_REPO_ROOT / reference).exists(),
              f"{notebook.name} references {reference}, which is not in the "
              "repository",
          )


class NotebookDeprecatedApiTest(unittest.TestCase):
  """No notebook may teach readers to create a deprecated artefact.

  The two directions are not the same, and conflating them would either miss
  the problem or break a correct notebook:

  `model.save_mmm` writes a new pickle. Nothing should do that -- the
  supported call is `meridian_serde.save_meridian`, and a tutorial that
  recommends otherwise leaves readers with artefacts in a format this fork is
  moving away from.

  `model.load_mmm` reads an existing one, which is the documented migration
  path in TRIAGE.md. `Meridian_Scenario_Planner_Beta.ipynb` uses it correctly,
  guarded by `if model_path.endswith('.pkl')` with `load_meridian` for
  everything else. That is allowed, and the guard is what makes it allowed.
  """

  def test_no_notebook_saves_with_the_deprecated_pickle_api(self):
    for notebook in _NOTEBOOKS:
      with self.subTest(notebook=notebook.name):
        self.assertFalse(
            _SAVE_MMM.search(_code(notebook)),
            f"{notebook.name} calls model.save_mmm, which writes a deprecated "
            "pickle. Use meridian_serde.save_meridian and a .binpb path.",
        )

  def test_any_deprecated_load_is_guarded_as_a_migration_path(self):
    for notebook in _NOTEBOOKS:
      code = _code(notebook)
      if not _LOAD_MMM.search(code):
        continue
      with self.subTest(notebook=notebook.name):
        self.assertIn(
            ".pkl",
            code,
            f"{notebook.name} calls model.load_mmm without any .pkl branch, "
            "so it is not reading a legacy artefact. Use "
            "meridian_serde.load_meridian.",
        )
        self.assertIn(
            "load_meridian",
            code,
            f"{notebook.name} calls model.load_mmm but never "
            "meridian_serde.load_meridian, so it offers no supported path.",
        )


if __name__ == "__main__":
  unittest.main()
