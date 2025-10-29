"""Space-compatible wrapper: run the existing dashboard script.

Hugging Face Spaces expects an `app.py` entrypoint. The existing
dashboard lives in `app/gradio-dashboard.py` (filename contains a dash so
it can't be imported as a module). We use runpy.run_path to execute the
script as __main__, which triggers the dashboard's launch block.
"""
import runpy


def main():
    # Execute the existing script as __main__ so its
    # `if __name__ == "__main__": dashboard.launch()` block runs.
    runpy.run_path("app/gradio-dashboard.py", run_name="__main__")


if __name__ == "__main__":
    main()
