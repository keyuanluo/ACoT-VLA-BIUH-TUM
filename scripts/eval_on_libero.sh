export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
source examples/libero/.venv/bin/activate

exp_name=test
task_name=${1}
port=${2}
resume_id=${3}

examples/libero/.venv/bin/python examples/libero/main.py \
    --args.task_suite_name=${task_name} \
    --args.exp_name=${exp_name} \
    --args.port=${port} \
    --args.resume_id=${resume_id}