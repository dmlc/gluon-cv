#!/bin/bash

# Run Jupyter in foreground if $JUPYTER_FG is set
if [[ "${JUPYTER_FG}" == "true" ]]; then
   jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token=''
   exit 0
else
   nohup jupyter-lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' > /dev/null 2>&1 &

   echo "Notebook server successfully started, a JupyterLab instance has been executed!"
   echo "Make local folders visible by volume mounting to /workspace/notebook"
   echo "To access visit http://localhost:8888 on your host machine."
   echo 'Ensure the following arguments to "docker run" are added to expose the server ports to your host machine:
      -p 8888:8888 -p 8787:8787 -p 8786:8786'
fi
