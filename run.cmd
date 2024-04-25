@echo off

cd src/generate_packet
del generate_packet.exe
mingw32-make
rem generate_packet.exe data.txt gold_sequence.txt gold_seq_end.txt data_bin.txt 
cd ../..
cd tests

rem python test_signal.py


python test_signal_SDR_v5.py

cd ./..



