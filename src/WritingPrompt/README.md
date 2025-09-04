# Creative Writing Task

## Story Generation

We provide a wrapper script for the story generation, which has the following mandatory commandline parameters.
```bash
write_stories.sh <total_range> <number_of_executions> <temperature> <output_name> <model_name>
```

## Story Evaluation

We provide a wrapper script for the story evaluation, which has the following mandatory commandline parameters.

```bash
evaluate_stories.sh <total_stories> <number_of_executions> <round_number> [dataset_type] [top_upvoted_csv] [eval_model_name] [binary_evaluation]
```

## Additional Functionalities

We provide a wrapper script to either the number of words of the short stories or delete them if their word count is not within certain bounds.
```bash
usage: clean_short_stories.py [-h] [--action {clean,count}] [--path PATH]

Clean up or count words in short stories from a file or directory.

optional arguments:
  -h, --help            show this help message and exit
  --action {clean,count}
                        The action to perform: 'clean' to delete short stories, 'count' to get word counts.
  --path PATH           The path to a file or directory to process.
```
