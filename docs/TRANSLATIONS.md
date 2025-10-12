# Translation Workflow

The Yam Image Processor uses Qt's translation system.  Application strings are
wrapped with `self.tr()` or `QCoreApplication.translate()` so that `.qm`
language packs can be loaded at runtime.

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
   * `yam_processor/i18n/yam_processor_<locale>.qm` – compiled language pack
     installed by the application at runtime.

   Set the `PYLUPDATE5` or `LRELEASE` environment variables to override the
   executables if they are not on your `PATH`.

3. Open the generated `.ts` files in Qt Linguist to translate strings.  Re-run
   the script to regenerate the `.qm` binaries after each update.

## Packaging

The compiled `.qm` files inside `yam_processor/i18n/` should be shipped with the
application distribution (wheels, PyInstaller bundle, etc.).

## Selecting a Language at Runtime

The bootstrap `AppCore` looks for translation packs inside
`yam_processor/i18n/`.  It tries locales provided by
`AppConfiguration.translation_locales` first.  When no explicit locale is
configured it falls back to the Qt system UI languages.  Example:

```python
from yam_processor import AppCore, AppConfiguration

config = AppConfiguration(translation_locales=("fr_FR",))
app_core = AppCore(config)
app_core.bootstrap()
```

Call `bootstrap()` after a `QApplication` has been created so the translation
packages can be installed on the running `QCoreApplication` instance.
