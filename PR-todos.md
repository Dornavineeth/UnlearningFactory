Curent evaluation configs and code does not distinguish between data splits.
- [x] restructure eval dump format 
- [x] rename yaml files for forget, update metrics and memorization files to fit new names
- [x] remove truth ratio dependence on paraphrased (access key)
- [x] fix truth ratio implementation to match TOFU
- [ ] create yaml files for other splits' metrics
- [ ] Implement metric aggregator for every metric {"overall":0.5, "details": {"index_to_probs": {}}}
- [ ] write truth ratio version for each split with averaging logic for different datasets in Truth Ratio