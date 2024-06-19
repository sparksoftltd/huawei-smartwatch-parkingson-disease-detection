RCLONE=rclone
SYNC=sync


### Input data
sync-input-activity-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/activity PD:huawei-parkingson-disease-analysis-detection/input/activity --config ./rclone.conf

sync-input-feature-extraction-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/feature/extraction PD:huawei-parkingson-disease-analysis-detection/input/feature/extraction --config ./rclone.conf

sync-input-feature-selection-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/feature/selection PD:huawei-parkingson-disease-analysis-detection/input/feature/selection --config .
	/rclone.conf

### Output data
sync-output-activity-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./output/activity PD:huawei-parkingson-disease-analysis-detection/output/activity --config ./rclone.conf

sync-output-feature-extraction-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./output/feature/extraction PD:huawei-parkingson-disease-analysis-detection/output/feature/extraction --config ./rclone.conf

sync-output-feature-selection-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./output/feature/selection PD:huawei-parkingson-disease-analysis-detection/output/feature/selection --config .
	/rclone.conf
