echo "Entrypoint: $1"
#python3 opti.py
xvfb-run -s "-screen 0 1400x900x24" python3 test_record.py $1
