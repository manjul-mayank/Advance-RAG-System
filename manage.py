import os
import sys
import platform
import multiprocessing # <-- Added import

if __name__ == "__main__":
    
    # --- FIX FOR MACOS (ARM-64) STABILITY ---
    # This addresses EXC_BAD_ACCESS (SIGSEGV) errors common in ML/numerical libraries
    # on macOS by forcing the safer 'spawn' multiprocessing start method.
    if platform.system() == "Darwin":
        try:
            # We use force=True to override any default settings
            multiprocessing.set_start_method("spawn", force=True) 
            # Note: A print statement here (like in the previous example) is useful 
            # for debugging but is optional in the final code.
        except RuntimeError:
            # Handle the case where set_start_method has already been called
            pass
    # ----------------------------------------
    
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_backend.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Activate your venv and install requirements."
        ) from exc
    execute_from_command_line(sys.argv)