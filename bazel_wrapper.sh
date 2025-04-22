 #!/bin/bash

 # TODO(chanwcom): Adds explanation

 PYTHONBIN=$(python -c "import sys; print(sys.executable)")
 PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")

 bazelisk-linux-amd64 "$@" \
   --python_path="$PYTHONBIN" \
   --test_env=PYTHONPATH="$PYTHONPATH"

