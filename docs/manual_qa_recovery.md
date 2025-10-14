# Manual QA: Autosave Recovery UI

The following checklist helps verify the autosave recovery workflow when
performing manual testing:

1. **Trigger an autosave error**
   * Start the application and perform enough edits to trigger an autosave.
   * Manually corrupt or lock the autosave file so the next autosave attempt
     fails (e.g., change permissions or remove the directory).
   * Observe that a structured error dialog appears. It should contain:
     - A status banner describing that crash markers were detected or that a
       recoverable snapshot is available.
     - A destructive “Discard autosave” action when a snapshot is present.
2. **Discard from the dialog**
   * Click “Discard autosave”. The dialog should close and the recovery manager
     should remove pending autosave artefacts from the workspace.
   * Confirm that the error dialog does not reappear on subsequent operations.
3. **Restore path messaging**
   * Allow the autosave to succeed again (fix the filesystem permissions and
     trigger another autosave).
   * Open a module error dialog via another recoverable error. The status
     banner should now highlight the location of the recoverable autosave
     snapshot without marking it as an error.
4. **Crash marker cleanup**
   * Perform a manual project save. After the save completes, confirm that the
     recovery status banner disappears from subsequent dialogs, demonstrating
     that crash markers were cleared successfully.

Document the results of this checklist in the test run report to confirm that
the UI is wired to the recovery manager callbacks.
