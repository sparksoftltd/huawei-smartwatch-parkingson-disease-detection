RCLONE=rclone
SYNC=sync

sync-activity-folder-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/activity PD:huawei-parkingson-disease-analysis-detection/activity --config ./rclone.conf

sync-feature-extraction-folder-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/feature/extraction PD:huawei-parkingson-disease-analysis-detection/feature/extraction --config ./rclone.conf

sync-feature-selection-folder-data:
	$(RCLONE) $(SYNC) -u -c --check-first ./input/feature/selection PD:huawei-parkingson-disease-analysis-detection/feature/selection --config .
	/rclone.conf


