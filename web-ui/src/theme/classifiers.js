/** Classifier roster — single source shared by the ledger and dashboard. */
export const CLASSIFIERS = [
  { id: 'rule_based', name: 'Rule-Based', type: 'rule' },
  { id: 'logistic_regression', name: 'Logistic Regression', type: 'ml' },
  { id: 'random_forest', name: 'Random Forest', type: 'ml' },
  { id: 'svm', name: 'SVM (RBF Kernel)', type: 'ml' },
  { id: 'naive_bayes', name: 'Naive Bayes', type: 'ml' },
];

/** API classification name → roster id. */
export const NAME_TO_ID = {
  'Rule-Based': 'rule_based',
  'Logistic Regression': 'logistic_regression',
  'Random Forest': 'random_forest',
  'SVM (RBF Kernel)': 'svm',
  'Naive Bayes': 'naive_bayes',
};
