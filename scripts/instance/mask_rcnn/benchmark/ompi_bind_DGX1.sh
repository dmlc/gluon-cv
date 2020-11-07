#!/bin/bash

case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    exec numactl --physcpubind=0-5,48-53 --membind=0 "${@}"
    ;;
1)
    exec numactl --physcpubind=6-11,54-59 --membind=0 "${@}"
    ;;
2)
    exec numactl --physcpubind=12-17,60-65 --membind=0 "${@}"
    ;;
3)
    exec numactl --physcpubind=18-23,66-71 --membind=0 "${@}"
    ;;
4)
    exec numactl --physcpubind=24-29,72-77 --membind=1 "${@}"
    ;;
5)
    exec numactl --physcpubind=30-35,78-83 --membind=1 "${@}"
    ;;
6)
    exec numactl --physcpubind=36-41,84-89 --membind=1 "${@}"
    ;;
7)
    exec numactl --physcpubind=42-47,90-95 --membind=1 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac

