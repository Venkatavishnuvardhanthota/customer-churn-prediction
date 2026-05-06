import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
def evaluate_model(model, X_test, y_test,
                   model_name="Model"):
    """
    Evaluates a trained ML model and returns
    all performance metrics as a dictionary.

    This is the MAIN evaluation function.
    Call this for any trained model!

    Parameters:
    -----------
    model : sklearn estimator
        Any trained sklearn model
        Example: RandomForestClassifier after .fit()
        Must have .predict() and .predict_proba()

    X_test : array-like or DataFrame
        Test features (NOT training features!)
        Shape: (n_samples, n_features)
        Must be SCALED if model was trained on scaled data!

    y_test : array-like or Series
        True labels for test set
        Shape: (n_samples,)
        Contains 0s and 1s

    model_name : str, optional
        Name of the model for display purposes
        Default: "Model"
        Example: "Random Forest" or "Logistic Regression"

    Returns:
    --------
    results : dict
        Dictionary containing all metric scores:
        {
            'model_name': str,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'auc': float,
            'predictions': array,
            'probabilities': array,
            'confusion_matrix': array
        }

    Example usage:
    --------------
    results = evaluate_model(
        rf_model,
        X_test_scaled,
        y_test,
        "Random Forest"
    )
    print(results['auc'])
    """

    # Generate Predictions

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    churn_proba = probabilities[:, 1]

    # Calculate All Metrics

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(y_test, churn_proba)

    cm = confusion_matrix(y_test, predictions)

    print(f"\n{'='*50}")
    print(f"  {model_name} — EVALUATION RESULTS")
    print(f"{'='*50}")

    print(f"\n{'Metric':<20} {'Score':>10} {'Meaning':>20}")
    print(f"{'─'*50}")

    metrics_display = [
        ('Accuracy',  accuracy,
         'Overall correct %'),
        ('Precision', precision,
         'Correct churn alerts'),
        ('Recall',    recall,
         'Churners caught ← KEY'),
        ('F1 Score',  f1,
         'Precision-recall balance'),
        ('AUC Score', auc,
         'Overall model quality'),
    ]

    for metric_name, value, meaning in metrics_display:

        print(f"{metric_name:<20} "
              f"{value:>10.4f} "
              f"{meaning:>20}")

    print(f"\n{'─'*50}")

    # Interpret AUC score automatically
    if auc >= 0.9:
        quality = "🌟 EXCELLENT"
    elif auc >= 0.8:
        quality = "✅ GOOD"
    elif auc >= 0.7:
        quality = "⚠️  ACCEPTABLE"
    else:
        quality = "❌ POOR — needs improvement"

    print(f"Model Quality: {quality}")
    print(f"{'='*50}\n")

    # Return All Results 

    results = {
        'model_name':       model_name,
        'accuracy':         accuracy,
        'precision':        precision,
        'recall':           recall,
        'f1':               f1,
        'auc':              auc,
        'predictions':      predictions,
        'probabilities':    churn_proba,
        'confusion_matrix': cm
    }

    return results


# ══════════════════════════════════════════════════
# FUNCTION 2: plot_confusion_matrix
# ══════════════════════════════════════════════════

def plot_confusion_matrix(results,
                          save_path=None):
    """
    Creates a beautiful confusion matrix heatmap.

    Parameters:
    -----------
    results : dict
        Output from evaluate_model() function
        Must contain 'confusion_matrix' and 'model_name'

    save_path : str, optional
        Full file path to save the chart image
        Default: None (don't save, just show)
        Example: '../outputs/confusion_matrix.png'

    Returns:
    --------
    fig : matplotlib Figure object
        The created figure
        Can be used in Streamlit with st.pyplot(fig)
    """

    cm = results['confusion_matrix']

    model_name = results['model_name']

    tn = cm[0, 0]

    fp = cm[0, 1]

    fn = cm[1, 0]

    tp = cm[1, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,

        annot=True,

        fmt='d',

        cmap='Blues',

        ax=ax,

        xticklabels=['No Churn', 'Churned'],

        yticklabels=['No Churn', 'Churned'],

        linewidths=2,

        linecolor='white',

        annot_kws={'size': 14, 'weight': 'bold'}
    )

    ax.set_title(
        f'{model_name}\nConfusion Matrix',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    ax.set_xlabel('Predicted Label', fontsize=12)

    ax.set_ylabel('Actual Label', fontsize=12)

    explained = (
        f'TN={tn:,}  FP={fp:,}  '
        f'FN={fn:,}  TP={tp:,}'
    )
    ax.text(
        0.5,
        -0.15,
        explained,
        transform=ax.transAxes,
        ha='center',
        fontsize=11
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )
        print(f"✅ Confusion matrix saved: {save_path}")

    return fig


# ══════════════════════════════════════════════════
# FUNCTION 3: plot_roc_curve
# ══════════════════════════════════════════════════

def plot_roc_curve(models_results,
                   save_path=None):
    """
    Plots ROC curves for one or multiple models.
    Used to visually compare model performance.

    WHAT IS ROC CURVE?
    → ROC = Receiver Operating Characteristic
    → A graph showing model performance at ALL
      possible threshold settings
    → X-axis = False Positive Rate (false alarms)
    → Y-axis = True Positive Rate (churners caught)
    → Diagonal line = random guessing (AUC=0.5)
    → Better model = curve closer to top-left corner
    → AUC = area under this curve

    Parameters:
    -----------
    models_results : list of dicts
        List of result dictionaries from evaluate_model()
        Can pass ONE or MULTIPLE models to compare!
        Example: [lr_results, rf_results]

    save_path : str, optional
        Path to save chart image
        Default: None

    Returns:
    --------
    fig : matplotlib Figure object
    """

    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#3498db', '#e74c3c', '#2ecc71',
              '#f39c12', '#9b59b6']

    for i, result in enumerate(models_results):

        fpr, tpr, _ = roc_curve(

            result['confusion_matrix'].ravel(),

            result['probabilities']
        )

        auc = result['auc']
        name = result['model_name']
        color = colors[i % len(colors)]

        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.5,
            label=f'{name} (AUC = {auc:.4f})'
        )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        linewidth=1.5,
        label='Random Guessing (AUC = 0.5)'
    )

    ax.set_xlabel('False Positive Rate\n'
                  '(False Alarms)',
                  fontsize=12)

    ax.set_ylabel('True Positive Rate\n'
                  '(Churners Caught)',
                  fontsize=12)

    ax.set_title('ROC Curve Comparison',
                 fontsize=14,
                 fontweight='bold')

    ax.legend(
        loc='lower right',
        fontsize=11
    )

    ax.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path,
                    dpi=150,
                    bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")

    return fig


# ══════════════════════════════════════════════════
# FUNCTION 4: compare_models
# ══════════════════════════════════════════════════

def compare_models(models_results):
    """
    Creates a comparison table of multiple models.
    Shows all metrics side by side to pick the winner!

    Parameters:
    -----------
    models_results : list of dicts
        List of result dictionaries from evaluate_model()
        Example: [lr_results, rf_results]

    Returns:
    --------
    comparison_df : pandas DataFrame
        Table with each model as column
        and each metric as row
        Easy to read and compare!

    Example:
    --------
    comparison = compare_models([lr_results, rf_results])
    display(comparison)  # shows in notebook
    """

    print("\n" + "="*60)
    print("  MODEL COMPARISON")
    print("="*60)

    comparison = {}

    for result in models_results:

        name = result['model_name']

        comparison[name] = {
            'Accuracy':  f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall':    f"{result['recall']:.4f}",
            'F1 Score':  f"{result['f1']:.4f}",
            'AUC Score': f"{result['auc']:.4f}"
        }

    comparison_df = pd.DataFrame(comparison)

    print(comparison_df.to_string())
    print(f"\n{'─'*60}")
    print("WINNERS BY METRIC:")
    print(f"{'─'*60}")

    metrics = ['Accuracy', 'Precision',
               'Recall', 'F1 Score', 'AUC Score']

    for metric in metrics:
        scores = {
            name: float(comparison[name][metric])
            for name in comparison
        }

        winner = max(scores, key=scores.get)

        best_score = scores[winner]
        print(f"  {metric:<15} → "
              f"{winner} ({best_score:.4f}) ✅")

    print(f"{'='*60}\n")

    return comparison_df


# ══════════════════════════════════════════════════
# FUNCTION 5: get_feature_importance
# ══════════════════════════════════════════════════

def get_feature_importance(model,
                           feature_names,
                           top_n=15,
                           save_path=None):
    """
    Extracts and visualizes feature importance
    from a trained Random Forest model.

    WHAT IS FEATURE IMPORTANCE?
    → A score for each feature showing how much
      it contributed to predictions
    → Higher score = more important for prediction
    → Helps understand WHAT drives churn!
    → Only works with tree-based models (Random Forest)

    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
        Must have .feature_importances_ attribute

    feature_names : list
        Names of all features in correct order
        Must match order used during training!

    top_n : int, optional
        How many top features to show
        Default: 15

    save_path : str, optional
        Path to save chart image

    Returns:
    --------
    importance_df : pandas DataFrame
        Feature names and importance scores
        Sorted from most to least important
    """

    if not hasattr(model, 'feature_importances_'):
        print("❌ This model doesn't support "
              "feature importance!")
        print("   Use Random Forest instead!")
        return None

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature':   feature_names,
        'Importance': importances
    })

    importance_df = importance_df\
        .sort_values(
            'Importance',
            ascending=False
        ).reset_index(drop=True)

    top_importance = importance_df.head(top_n)

    print(f"\nTOP {top_n} MOST IMPORTANT FEATURES:")
    print("─" * 45)
    for _, row in top_importance.iterrows():

        bar = '█' * int(row['Importance'] * 200)

        print(f"  {row['Feature']:<35} "
              f"{row['Importance']:.4f} {bar}")

    # Visualization

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        '#e74c3c' if imp > 0.05
        else '#f39c12' if imp > 0.02
        else '#3498db'
        for imp in top_importance['Importance']
    ]

    bars = ax.barh(
        top_importance['Feature'],

        top_importance['Importance'],

        color=colors,
        edgecolor='white',
        linewidth=0.5,
        height=0.7
    )
    for bar, value in zip(
        bars,
        top_importance['Importance']
    ):
        ax.text(
            bar.get_width() + 0.001,

            bar.get_y() + bar.get_height() / 2,

            f'{value:.4f}',
            va='center',
            fontsize=9
        )

    ax.set_title(
        f'Top {top_n} Most Important Features\n'
        f'for Churn Prediction',
        fontsize=14,
        fontweight='bold'
    )

    ax.set_xlabel('Feature Importance Score',
                  fontsize=12)

    ax.set_ylabel('Feature', fontsize=12)

    ax.invert_yaxis()


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path,
                    dpi=150,
                    bbox_inches='tight')
        print(f"✅ Feature importance saved: {save_path}")

    return importance_df, fig


# ══════════════════════════════════════════════════
# FUNCTION 6: full_evaluation_report
# ══════════════════════════════════════════════════

def full_evaluation_report(models_dict,
                           X_test,
                           y_test,
                           feature_names,
                           save_dir='../outputs/'):
    """
    Runs COMPLETE evaluation for multiple models.
    This is the MASTER evaluation function.
    Calls all other functions automatically!

    Parameters:
    -----------
    models_dict : dict
        Dictionary of model name → trained model
        Example: {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model
        }

    X_test : array-like
        Scaled test features

    y_test : array-like
        True test labels

    feature_names : list
        Column names for feature importance

    save_dir : str, optional
        Directory to save all chart images
        Default: '../outputs/'

    Returns:
    --------
    all_results : dict
        All results for all models
    best_model_name : str
        Name of the best performing model
    """

    print("=" * 60)
    print("  RUNNING FULL EVALUATION REPORT")
    print("=" * 60)

    all_results = {}

    # Evaluate Each Model 
    for model_name, model in models_dict.items():

        results = evaluate_model(
            model,
            X_test,
            y_test,
            model_name
        )
        all_results[model_name] = results

    results_list = list(all_results.values())

    comparison_df = compare_models(results_list)

    # Plot Confusion Matrix for Each Model 
    for model_name, results in all_results.items():
        save_path = (
            f"{save_dir}"
            f"eval_confusion_{model_name.lower().replace(' ', '_')}.png"
        )

        plot_confusion_matrix(results, save_path)

    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            save_path = (
                f"{save_dir}"
                f"eval_importance_{model_name.lower().replace(' ', '_')}.png"
            )
            get_feature_importance(
                model,
                feature_names,
                top_n=15,
                save_path=save_path
            )

    best_model_name = max(
        all_results,
        key=lambda name: all_results[name]['auc']
    )

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   AUC: {all_results[best_model_name]['auc']:.4f}")

    return all_results, best_model_name
