# after running the profiling, run this to generate the .csv file from the nsys-rep output
nsys export --type sqlite --output mlora_profile.sqlite mlora_profile.nsys-rep --force-overwrite true
python scripts/performance_report.py --db mlora_profile.sqlite --output mlora_performance_report.csv