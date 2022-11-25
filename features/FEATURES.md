# Setting up the feature directory

All our generated features are saved into the `./features` directory. We describe the three different sets of feature sets that are generated:

* **Text classifier weights**: For each dataset, we use three different prompt types for generating the classifier weights in our experiments using CLIP's text encoder: ensemble (from the ensemble prompts), cupl (from the customised GPT-3 prompts) and combined (from the concatenated cupl+ensemble prompts). Their naming is in the format: `<dataset>_zeroshot_text_weights_m<backbone>_pt<prompt_type>.pt`.
* **Test/Validation features**: These are the test and validation set features along with their labels encoded using CLIP's image encoder. Their naming is in the format: `<dataset>_f_<test/val>_m<backbone>.pt` for features and `<dataset>_t_<test/val>_m<backbone>.pt` for labels.
* **SuS features**: These are the support set features (either using SuS-LC or SuS-SD) along with their labels encoded using CLIP's image encoder. Their naming is in the format: `sus_<sus_type>_<prompting_strategy>_<dataset>_f_m<backbone>.pt` for features and `sus_<sus_type>_<prompting_strategy>_<dataset>_t_m<backbone>.pt` for labels. `<sus_type>` is `lc` for SuS-LC and `sd` for SuS-SD. `<prompting_strategy>` is `photo` for *Photo* prompting strategy and `cupl` for *CuPL* prompting strategy. Refer to Sec. 3.1 of paper for details.

