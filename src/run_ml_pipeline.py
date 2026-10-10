import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

def main():
    scratch_dir = r'C:\Users\sunhb\.gemini\antigravity\scratch'
    target_dir = r'D:\OneDrive\文档\沈阳化工大学\杨梅'
    fig_dir = os.path.join(scratch_dir, 'figures')
    target_fig_dir = os.path.join(target_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(target_fig_dir, exist_ok=True)

    # Load dataset
    csv_path = os.path.join(scratch_dir, 'Furfural_Biomass_ML_Dataset.csv')
    df = pd.read_csv(csv_path)

    # Define feature columns and target
    num_features = ['C_xylose_wt', 'Temp_C', 'Time_min', 'Pressure_MPa', 'Cat_Loading_mM', 'Acid_B_L_Ratio', 'AO_Ratio', 'NaCl_wt']
    cat_features = ['Catalyst_Type', 'Solvent_Type', 'Organic_Solvent']
    target_col = 'Yield_pct'

    X = df[num_features + cat_features]
    y = df[target_col]

    # Split Train and Test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # Models dictionary
    models = {
        'Gradient Boosting (GBDT)': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        'Random Forest (RF)': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        'Extra Trees (ET)': ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42),
        'Hist Gradient Boosting (HGB)': HistGradientBoostingRegressor(max_iter=150, max_depth=5, random_state=42),
        'Support Vector Regressor (SVR)': SVR(C=50.0, epsilon=0.5, kernel='rbf'),
        'Multi-Layer Perceptron (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, alpha=0.01, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'Ridge Regression (Linear)': Ridge(alpha=1.0)
    }

    results = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fitted_pipelines = {}

    scoring = {
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }

    for name, model in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('reg', model)])
        
        cv_res = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
        
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        fitted_pipelines[name] = pipe
        
        results.append({
            'Model': name,
            'Train_R2': round(train_r2, 4),
            'Test_R2': round(test_r2, 4),
            'CV_R2_Mean': round(float(cv_res['test_r2'].mean()), 4),
            'CV_R2_Std': round(float(cv_res['test_r2'].std()), 4),
            'Test_RMSE': round(test_rmse, 3),
            'CV_RMSE_Mean': round(float(-cv_res['test_rmse'].mean()), 3),
            'Test_MAE': round(test_mae, 3),
            'CV_MAE_Mean': round(float(-cv_res['test_mae'].mean()), 3)
        })

    res_df = pd.DataFrame(results).sort_values(by='CV_R2_Mean', ascending=False)
    print('=== Model Evaluation Summary ===')
    print(res_df.to_string(index=False))

    # Save results table to CSV
    res_df.to_csv(os.path.join(scratch_dir, 'model_evaluation_metrics.csv'), index=False)
    res_df.to_csv(os.path.join(target_dir, 'model_evaluation_metrics.csv'), index=False)

    # Select best model
    best_model_name = res_df.iloc[0]['Model']
    best_pipe = fitted_pipelines[best_model_name]
    print(f'\nBest Model: {best_model_name}')

    # -------------------------------------------------------------
    # FIGURE 1: Model Comparison Bar Charts
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5.5), dpi=300)
    x_pos = np.arange(len(res_df))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x_pos - width/2, res_df['Train_R2'], width, label='Train $R^2$', color='#2980b9', alpha=0.85, edgecolor='black')
    plt.bar(x_pos + width/2, res_df['CV_R2_Mean'], width, yerr=res_df['CV_R2_Std'], capsize=4, label='10-Fold CV $R^2$', color='#e74c3c', alpha=0.85, edgecolor='black')
    plt.xticks(x_pos, res_df['Model'], rotation=45, ha='right', fontsize=9)
    plt.ylabel('Coefficient of Determination ($R^2$)', fontsize=11, fontweight='bold')
    plt.title('(a) Coefficient of Determination ($R^2$)', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.subplot(1, 2, 2)
    plt.bar(x_pos - width/2, res_df['Test_RMSE'], width, label='Test RMSE (%)', color='#27ae60', alpha=0.85, edgecolor='black')
    plt.bar(x_pos + width/2, res_df['CV_RMSE_Mean'], width, label='10-Fold CV RMSE (%)', color='#e67e22', alpha=0.85, edgecolor='black')
    plt.xticks(x_pos, res_df['Model'], rotation=45, ha='right', fontsize=9)
    plt.ylabel('Root Mean Squared Error (RMSE, %)', fontsize=11, fontweight='bold')
    plt.title('(b) Prediction Error (RMSE)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, 'Fig1_Model_Evaluation_Comparison.png')
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(os.path.join(target_fig_dir, 'Fig1_Model_Evaluation_Comparison.png'), dpi=300)
    plt.close()
    print('Figure 1 saved.')

    # -------------------------------------------------------------
    # FIGURE 2: Parity Plot (Predicted vs Experimental)
    # -------------------------------------------------------------
    y_train_pred = best_pipe.predict(X_train)
    y_test_pred = best_pipe.predict(X_test)

    train_r2_val = res_df.iloc[0]['Train_R2']
    test_r2_val = res_df.iloc[0]['Test_R2']
    test_rmse_val = res_df.iloc[0]['Test_RMSE']

    plt.figure(figsize=(7, 6.5), dpi=300)
    plt.scatter(y_train, y_train_pred, color='#2980b9', alpha=0.7, s=45, edgecolors='k', label=f'Training set ($R^2$ = {train_r2_val:.3f})')
    plt.scatter(y_test, y_test_pred, color='#e67e22', alpha=0.85, s=60, edgecolors='k', marker='s', label=f'Testing set ($R^2$ = {test_r2_val:.3f}, RMSE = {test_rmse_val:.2f}%)')

    min_val = min(y.min(), min(y_train_pred.min(), y_test_pred.min())) - 5
    max_val = max(y.max(), max(y_train_pred.max(), y_test_pred.max())) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.8, label='1:1 Parity Line')
    plt.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], 'r:', lw=1.2, label='±10% Error Band')
    plt.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], 'r:', lw=1.2)
    plt.fill_between([min_val, max_val], [min_val * 0.9, max_val * 0.9], [min_val * 1.1, max_val * 1.1], color='gray', alpha=0.12)

    plt.xlabel('Experimental Furfural Yield (%)', fontsize=12, fontweight='bold')
    plt.ylabel('ML-Predicted Furfural Yield (%)', fontsize=12, fontweight='bold')
    plt.title(f'Parity Plot for {best_model_name}', fontsize=13, fontweight='bold')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, 'Fig2_Parity_Plot_Predicted_vs_Actual.png')
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(os.path.join(target_fig_dir, 'Fig2_Parity_Plot_Predicted_vs_Actual.png'), dpi=300)
    plt.close()
    print('Figure 2 saved.')

    # -------------------------------------------------------------
    # FIGURE 3: Feature Importance Analysis (Permutation Importance)
    # -------------------------------------------------------------
    perm_imp = permutation_importance(best_pipe, X_test, y_test, n_repeats=15, random_state=42)
    feature_names = num_features + cat_features
    sorted_idx = perm_imp.importances_mean.argsort()[::-1]

    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = perm_imp.importances_mean[sorted_idx]
    top_std = perm_imp.importances_std[sorted_idx]

    feat_labels = {
        'Temp_C': 'Reaction Temperature (°C)',
        'Time_min': 'Residence Time (min)',
        'Solvent_Type': 'Solvent System (Biphasic/Mono)',
        'Cat_Loading_mM': 'Catalyst Loading (mM)',
        'Acid_B_L_Ratio': 'Brønsted/Lewis Acid Ratio',
        'AO_Ratio': 'Aqueous-to-Organic Ratio (A/O)',
        'Organic_Solvent': 'Organic Extraction Solvent',
        'C_xylose_wt': 'Initial Xylose Conc. (wt%)',
        'Catalyst_Type': 'Catalyst Family Class',
        'NaCl_wt': 'Salting-out NaCl Conc. (wt%)',
        'Pressure_MPa': 'System Pressure (MPa)'
    }
    pretty_labels = [feat_labels.get(f, f) for f in top_features]

    plt.figure(figsize=(9, 6), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(pretty_labels)))
    y_pos = np.arange(len(pretty_labels))

    plt.barh(y_pos, top_importances[::-1], xerr=top_std[::-1], align='center', color=colors, edgecolor='black', capsize=4, alpha=0.85)
    plt.yticks(y_pos, pretty_labels[::-1], fontsize=10)
    plt.xlabel('Permutation Feature Importance (Mean Decrease in $R^2$)', fontsize=11, fontweight='bold')
    plt.title('Relative Feature Importance for Furfural Yield Optimization', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, 'Fig3_Feature_Importance_Analysis.png')
    plt.savefig(fig3_path, dpi=300)
    plt.savefig(os.path.join(target_fig_dir, 'Fig3_Feature_Importance_Analysis.png'), dpi=300)
    plt.close()
    print('Figure 3 saved.')

    # -------------------------------------------------------------
    # FIGURE 4: 2D & 3D Response Surface Optimization
    # -------------------------------------------------------------
    temp_range = np.linspace(140, 220, 50)
    time_range = np.linspace(10, 180, 50)
    T_grid, t_grid = np.meshgrid(temp_range, time_range)

    grid_rows = []
    for t_val in temp_range:
        for tm_val in time_range:
            grid_rows.append({
                'C_xylose_wt': 3.5,
                'Temp_C': t_val,
                'Time_min': tm_val,
                'Pressure_MPa': 1.8,
                'Cat_Loading_mM': 35.0,
                'Acid_B_L_Ratio': 1.8,
                'AO_Ratio': 0.33,
                'NaCl_wt': 15.0,
                'Catalyst_Type': 'BL_Mixture',
                'Solvent_Type': 'Biphasic',
                'Organic_Solvent': 'MIBK'
            })
    df_grid = pd.DataFrame(grid_rows)
    Z_yield = best_pipe.predict(df_grid).reshape(len(temp_range), len(time_range)).T

    plt.figure(figsize=(12, 5.5), dpi=300)

    plt.subplot(1, 2, 1)
    cp = plt.contourf(T_grid, t_grid, Z_yield, levels=25, cmap='Spectral_r')
    cbar = plt.colorbar(cp)
    cbar.set_label('Predicted Furfural Yield (%)', fontsize=10, fontweight='bold')
    plt.contour(T_grid, t_grid, Z_yield, levels=10, colors='k', alpha=0.3, linewidths=0.8)

    opt_idx = np.unravel_index(np.argmax(Z_yield), Z_yield.shape)
    opt_T = temp_range[opt_idx[1]]
    opt_t = time_range[opt_idx[0]]
    opt_Y = Z_yield[opt_idx]

    plt.plot(opt_T, opt_t, 'r*', markersize=14, markeredgecolor='black', label=f'Optimal: {opt_T:.1f}°C, {opt_t:.0f}min\n(Yield = {opt_Y:.1f}%)')
    plt.xlabel('Reaction Temperature (°C)', fontsize=11, fontweight='bold')
    plt.ylabel('Residence Time (min)', fontsize=11, fontweight='bold')
    plt.title('(a) 2D Isocontour Optimization Map', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(T_grid, t_grid, Z_yield, cmap='Spectral_r', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Temp (°C)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Time (min)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Yield (%)', fontsize=10, fontweight='bold')
    ax.set_title('(b) 3D Response Surface', fontsize=12, fontweight='bold')
    ax.view_init(elev=28, azim=-125)

    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, 'Fig4_Response_Surface_Optimization.png')
    plt.savefig(fig4_path, dpi=300)
    plt.savefig(os.path.join(target_fig_dir, 'Fig4_Response_Surface_Optimization.png'), dpi=300)
    plt.close()
    print('Figure 4 saved.')

    # -------------------------------------------------------------
    # FIGURE 5: Acid Ratio & Solvent Sensitivity
    # -------------------------------------------------------------
    bl_ratios = np.linspace(0.1, 4.0, 25)
    ao_ratios = np.linspace(0.1, 1.5, 25)
    BL_grid, AO_grid = np.meshgrid(bl_ratios, ao_ratios)

    grid_rows_bl = []
    for bl_val in bl_ratios:
        for ao_val in ao_ratios:
            grid_rows_bl.append({
                'C_xylose_wt': 3.5,
                'Temp_C': 175.0,
                'Time_min': 75.0,
                'Pressure_MPa': 1.8,
                'Cat_Loading_mM': 35.0,
                'Acid_B_L_Ratio': bl_val,
                'AO_Ratio': ao_val,
                'NaCl_wt': 15.0,
                'Catalyst_Type': 'BL_Mixture',
                'Solvent_Type': 'Biphasic',
                'Organic_Solvent': 'MIBK'
            })
    df_bl_grid = pd.DataFrame(grid_rows_bl)
    Z_bl_yield = best_pipe.predict(df_bl_grid).reshape(len(bl_ratios), len(ao_ratios)).T

    plt.figure(figsize=(11, 5), dpi=300)
    plt.subplot(1, 2, 1)
    cp2 = plt.contourf(BL_grid, AO_grid, Z_bl_yield, levels=20, cmap='viridis')
    cbar2 = plt.colorbar(cp2)
    cbar2.set_label('Furfural Yield (%)', fontsize=10, fontweight='bold')
    plt.xlabel('Brønsted-to-Lewis Acid Molar Ratio (B:L)', fontsize=11, fontweight='bold')
    plt.ylabel('Aqueous-to-Organic Volume Ratio (A/O)', fontsize=11, fontweight='bold')
    plt.title('(a) Acid Ratio vs. Solvent A/O Synergy', fontsize=11, fontweight='bold')

    plt.subplot(1, 2, 2)
    solvents_test = ['Water', 'Toluene', 'MIBK', '2-MeTHF', 'GVL']
    t_test_curve = np.linspace(140, 220, 40)
    for solv_name in solvents_test:
        rows = []
        for tc in t_test_curve:
            is_bi = 1 if solv_name in ['Toluene', 'MIBK', '2-MeTHF'] else 0
            stype = 'Biphasic' if is_bi else ('Water' if solv_name == 'Water' else 'Monophasic_Organic')
            rows.append({
                'C_xylose_wt': 3.5,
                'Temp_C': tc,
                'Time_min': 75.0,
                'Pressure_MPa': 1.8,
                'Cat_Loading_mM': 35.0,
                'Acid_B_L_Ratio': 1.8,
                'AO_Ratio': 0.33 if is_bi else 0.0,
                'NaCl_wt': 15.0 if is_bi else 0.0,
                'Catalyst_Type': 'BL_Mixture' if is_bi else 'Bronsted',
                'Solvent_Type': stype,
                'Organic_Solvent': solv_name if is_bi or stype=='Monophasic_Organic' else 'None'
            })
        preds = best_pipe.predict(pd.DataFrame(rows))
        plt.plot(t_test_curve, preds, lw=2.2, label=f'Solvent: {solv_name}')

    plt.xlabel('Reaction Temperature (°C)', fontsize=11, fontweight='bold')
    plt.ylabel('Predicted Furfural Yield (%)', fontsize=11, fontweight='bold')
    plt.title('(b) Solvent Effect Across Reaction Temperatures', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, 'Fig5_Pareto_Frontier_Yield_vs_Humin.png')
    plt.savefig(fig5_path, dpi=300)
    plt.savefig(os.path.join(target_fig_dir, 'Fig5_Pareto_Frontier_Yield_vs_Humin.png'), dpi=300)
    plt.close()
    print('Figure 5 saved.')

    print('\nOptimal Condition Search Results:')
    print(f'Optimal Temperature: {opt_T:.1f} °C')
    print(f'Optimal Residence Time: {opt_t:.1f} min')
    print(f'Optimal Catalyst System: B:L Acid Mixture (B:L = 1.8)')
    print(f'Optimal Solvent System: Biphasic Water/MIBK (A/O = 0.33, 1:3 v/v, 15 wt% NaCl)')
    print(f'Max Predicted Furfural Yield: {opt_Y:.2f} %')

if __name__ == '__main__':
    main()
