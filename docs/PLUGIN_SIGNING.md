# Plugin Signing Workflow

To protect the plugin supply chain the application now validates detached
signatures for every module that is discovered during startup. Plugin authors
must ship both the Python source file and a signature file alongside it so that
the loader can authenticate the code before it is executed.

## Trust store management

* Trusted signing keys are provided to the application via the
  `AppConfiguration.plugin_trust_store_paths` setting. Each file listed in this
  collection is expected to contain one or more PEM encoded public keys or X.509
  certificates.
* Keys can be rotated by updating the trust store files – the
  `ModuleSignatureVerifier` helper automatically loads every usable RSA public
  key that appears in the PEM data.
* The verifier currently supports RSA keys that are compatible with
  PKCS#1 v1.5 signatures over SHA-256 digests.

## Generating signatures

Plugin publishers should sign every module file before distributing it:

1. Generate a key pair. The example below uses OpenSSL to create a 3072-bit
   RSA key and extract the corresponding public key for the trust store.

   ```bash
   openssl genrsa -out plugin_signer.key 3072
   openssl rsa -in plugin_signer.key -pubout -out plugin_signer.pub
   ```

2. Add `plugin_signer.pub` (or an equivalent certificate) to a trust store file
   that is referenced from the application's configuration.

3. Create a detached signature for each module file using the private key. The
   resulting `.sig` file must be placed next to the module so that the loader
   can find it. When the module file has a suffix (for example `module.py`) the
   expected signature name is `module.py.sig`.

   ```bash
   openssl dgst -sha256 -sign plugin_signer.key -out module.py.sig module.py
   ```

4. Ship the module and its signature together. If the signature is missing or
   does not validate against the configured trust store the plugin will be
   skipped during discovery.

## Loader behaviour summary

* The loader refuses to execute any plugin module whose signature cannot be
  validated against the configured trust store.
* Missing signature files, empty signatures, and invalid signatures all cause
  the module to be skipped. Failures are logged with the module name and the
  signature path to aid debugging.
* When a module passes verification, discovery proceeds as before and the
  module's `register_module` hook is invoked if it is present.

