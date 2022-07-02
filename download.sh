wget https://lodmedia.hb.bizmrg.com/case_files/768820/train_dataset_train.zip
wget https://lodmedia.hb.bizmrg.com/case_files/768820/test_dataset_test.zip

unzip -q train_dataset_train.zip
mkdir test
unzip -q -d test/img test_dataset_test.zip
mkdir test/json
mkdir submit