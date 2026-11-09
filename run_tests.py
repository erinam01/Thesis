#### VALIDATION FUNCTIONS ####

from pathlib import Path
import sys
import unittest

## PUT THEM INTO A REPORT INSTEAD OF PRINTING


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parent

    # Find all test_*.py files inside the tests folder.
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(project_folder / "tests"),
        pattern="test_*.py"
    )

    # Run them and print a result for each test.
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return a failure status if any tests failed or raised errors.
    sys.exit(0 if result.wasSuccessful() else 1)