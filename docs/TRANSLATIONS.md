# Translation Workflow

The Yam Image Processor uses Qt's translation system.  Application strings are
wrapped with `self.tr()` or `QCoreApplication.translate()` so that `.qm`
language packs can be loaded at runtime.  Runtime loading is handled by
`core.i18n.TranslationLoader`, which installs any compiled catalogues placed in
the top-level `translations/` directory before windows are created.

## Requirements

* Qt Linguist tools (`pylupdate5` and `lrelease`).  They ship with PyQt5 and Qt
  SDK installations.  On Debian/Ubuntu systems you can install them with
  `apt-get install pyqt5-dev-tools`.

## Generating or Updating Catalogues

1. Choose the locales you want to generate.  Locale identifiers can use either
   `ll` or `ll_CC` (language + territory) format.  The runtime loader normalises
   dashes (`en-US`) to underscores (`en_US`).
2. Run the build script from the project root, passing each target locale:

   ```bash
   scripts/build_translations.sh en_US fr
   ```

   * `translations/yam_processor_<locale>.ts` – editable catalogue for Qt
     Linguist.
   * `translations/yam_processor_<locale>.qm` – compiled language pack loaded
     by the desktop entry points.

   Set the `PYLUPDATE5` or `LRELEASE` environment variables to override the
   executables if they are not on your `PATH`.

3. Open the generated `.ts` files in Qt Linguist to translate strings.  Re-run
   the script to regenerate the `.qm` binaries after each update.

## Packaging

Ship the `.qm` files inside the repository-level `translations/` directory with
any binary distribution (wheels, PyInstaller bundle, etc.).

## Selecting a Language at Runtime

The lightweight `core.AppCore` exposes configuration flags that control the
loader.  Override them before calling the module entry points if you want to
force a specific locale or provide additional lookup directories:

```python
from core.app_core import AppConfiguration, AppCore

config = AppConfiguration(
    translation_locales=("fr_FR",),
    translation_directories=["/opt/yam/translations", "./translations"],
)
app_core = AppCore(config)
app_core.ensure_bootstrapped()
```

The entry points (`preprocessing22`, `segmentation25`, `extraction18`) call
`core.i18n.bootstrap_translations()` once a `QApplication` has been created so
the language packs are installed before any top-level windows appear.
