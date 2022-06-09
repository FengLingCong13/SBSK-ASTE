local module_initializer = [
    [".*weight", {"type": "xavier_normal"}],
    [".*weight_matrix", {"type": "xavier_normal"}]];
local template = {
  SpanModel: {
    local span_model = self,

    // Mapping from target task to the metric used to assess performance on that task.
    local validation_metrics = {
      'ner': '+MEAN__ner_f1',
      'relation': '+MEAN__relation_f1',
    },

    ////////////////////

    // REQUIRED VALUES. Must be set by child class.

    // Paths to train, dev, and test data.
    data_paths :: error 'Must override `data_paths`',

    // Weights on the losses of the model components (e.g. NER, relation, etc).
    loss_weights :: error 'Must override `loss_weights`',

    // Make early stopping decisions based on performance for this task.
    // Options are: `['ner', 'relation']`
    target_task :: error 'Must override `target_task`',

    // DEFAULT VALUES. May be set by child class..
    bert_model :: 'bert-base-cased',
    // If using a different BERT, this number may be different. It's up to the user to set the
    // appropriate value.
    max_wordpieces_per_sentence :: 512,
    max_span_width :: 8,
    cuda_device :: -1,

    ////////////////////

    // All remaining values can be overridden using the `:+` mechanism
    // described in `doc/config.md`
    random_seed: 13370,
    numpy_seed: 1337,
    pytorch_seed: 133,
    dataset_reader: {
      type: 'span_model',
      token_indexers: {
        bert: {
          type: 'pretrained_transformer_mismatched',
          model_name: span_model.bert_model,
          max_length: span_model.max_wordpieces_per_sentence
        },
      },
      max_span_width: span_model.max_span_width
    },
    train_data_path: span_model.data_paths.train,
    validation_data_path: span_model.data_paths.validation,
    test_data_path: span_model.data_paths.test,
    // If provided, use pre-defined vocabulary. Else compute on the fly.
    model: {
      type: 'span_model',
      embedder: {
        token_embedders: {
          bert: {
            type: 'pretrained_transformer_mismatched',
            model_name: span_model.bert_model,
            max_length: span_model.max_wordpieces_per_sentence
          },
        },
      },
      initializer: {  // Initializer for shared span representations.
        regexes:
          [['_span_width_embedding.weight', { type: 'xavier_normal' }]],
      },
      module_initializer: {  // Initializer for component module weights.
        regexes:
          [
            ['.*weight', { type: 'xavier_normal' }],
            ['.*weight_matrix', { type: 'xavier_normal' }],
          ],
      },
      loss_weights: span_model.loss_weights,
      feature_size: 128,
      max_span_width: span_model.max_span_width,
      target_task: span_model.target_task,
      feedforward_params: {
        num_layers: 2,
        hidden_dims: 150,
        dropout: 0.4,
      },
      modules: {
        ner: {},
        relation: {
          spans_per_word: 0.5,
        },
      },
    },
    data_loader: {
      sampler: {
       type: "random",
     },
    },
    trainer: {
      checkpointer: {
        num_serialized_models_to_keep: 3,
      },
      num_epochs: 50,
      grad_norm: 5.0,
      cuda_device: span_model.cuda_device,
      validation_metric: validation_metrics[span_model.target_task],
      optimizer: {
        type: 'adamw',
        lr: 1e-3,
        weight_decay: 0.0,
        parameter_groups: [
          [
            ['_embedder'],
            {
              lr: 5e-5,
              weight_decay: 0.01,
              finetune: true,
            },
          ],
        ],
      },
      learning_rate_scheduler: {
        type: 'slanted_triangular'
      }
    },
  },
};

template.SpanModel {
  bert_model: "bert-base-uncased",
  cuda_device: 0,
  data_paths: {
    train: "/tmp/train.json",
    validation: "/tmp/dev.json",
    test: "/tmp/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
  },
  model +: {
    modules +: {
      relation: {
        spans_per_word: 0.5,
        use_distance_embeds: false,
        use_ner_scores_for_prune: false,
        use_span_pair_aux_task: false,
        use_span_pair_aux_task_after_prune: false,
        use_pruning: true,
        use_pair_feature_multiply: false,
        use_pair_feature_maxpool: false,
        use_pair_feature_cls: false,
        use_span_loss_for_pruners: false,
        use_ope_down_project: false,
        use_pairwise_down_project: false,
        use_classify_mask_pruner: false,
        use_bi_affine_classifier: false,
        use_bi_affine_pruner: false,
        neg_class_weight: -1,
        use_focal_loss: false,
        focal_loss_gamma: 2,
        span_length_loss_weight_gamma: 0.0,
        use_bag_pair_scorer: false,
        use_bi_affine_v2: false,
        use_single_pool: false,
      },
      ner: {
        use_bi_affine: false,
        neg_class_weight: -1,
        use_focal_loss: false,
        focal_loss_gamma: 2,
        use_double_scorer: false,
        use_gold_for_train_prune_scores: false,
        use_single_pool: false,
      },
      gat_tree: {
        span_emb_dim: 768,
        tree_prop: 2,
        tree_dropout: 0.4,
        feature_dim: 20,
        aggcn_heads: 4,
        aggcn_sublayer_first: 2,
        aggcn_sublayer_second: 4,
      },
    },
    use_ner_embeds: false,
    span_extractor_type: "endpoint",
    relation_head_type: "new",
    use_span_width_embeds: true,
    use_bilstm_after_embedder: false,

    use_double_mix_embedder: false,
//    use_double_mix_embedder: true,
//    embedder +: {
//      token_embedders +: {
//        bert +: {
//          type: 'double_mix_ptm',
//        },
//      },
//    },

//    feature_size: 50,
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 10,  # Set to < 5 for quick debugging
    optimizer: {
      type: 'adamw',
      lr: 1e-3,
      weight_decay: 0.0,
      parameter_groups: [
        [
          ['_matched_embedder'],  # May need to switch if using different embedder type in future
//          ['_embedder'],
          {
            lr: 5e-5,
            weight_decay: 0.01,
            finetune: true,
          },
        ],
        [
          ['scalar_parameters'],
          {
            lr: 1e-2,
          },
        ],
      ],
    }
  },
}
