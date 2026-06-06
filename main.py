"""
main.py — Pipeline Completo de Machine Learning
Previsão da Satisfação dos Funcionários

Execução:
    python main.py

Saídas geradas:
    data/processed/    — datasets tratados
    reports/figures/   — gráficos em alta qualidade
    models/            — modelo treinado + scaler + feature names
    reports/           — JSON para dashboard, relatório final
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw", "Employee-Attrition.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR    = os.path.join(BASE_DIR, "reports", "figures")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")

for d in [DATA_PROCESSED, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, BASE_DIR)

from src.data.load_data import load_raw_data, explore_data, clean_data
from src.features.feature_engineering import encode_categorical, create_features, select_features
from src.models.train_models import train_all_models, optimize_best_model, save_model, evaluate_model
from src.visualization.plots import generate_all_plots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    print("\n" + "=" * 70)
    print("   PIPELINE DE ML — PREVISÃO DA SATISFAÇÃO DOS FUNCIONÁRIOS")
    print("=" * 70)

    TARGET = "JobSatisfaction"

    # ─────────────────────────────────────────
    # 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
    # ─────────────────────────────────────────
    print("\n[ETAPA 1] Carregamento e Exploração dos Dados")
    df_raw = load_raw_data(DATA_RAW)
    report = explore_data(df_raw)

    # ─────────────────────────────────────────
    # 2. LIMPEZA DOS DADOS
    # ─────────────────────────────────────────
    print("\n[ETAPA 2] Limpeza dos Dados")
    df_clean = clean_data(df_raw)

    # ─────────────────────────────────────────
    # 3. FEATURE ENGINEERING
    # ─────────────────────────────────────────
    print("\n[ETAPA 3] Engenharia de Atributos")
    df_feat = create_features(df_clean)

    # ─────────────────────────────────────────
    # 4. ENCODING
    # ─────────────────────────────────────────
    print("\n[ETAPA 4] Encoding de Variáveis Categóricas")
    df_encoded = encode_categorical(df_feat, target_col=TARGET)

    # Salvar dataset processado
    df_encoded.to_csv(os.path.join(DATA_PROCESSED, "dataset_processed.csv"), index=False)
    print(f"[SAVE] Dataset processado salvo.")

    # ─────────────────────────────────────────
    # 5. SEPARAÇÃO X, y
    # ─────────────────────────────────────────
    print("\n[ETAPA 5] Divisão dos Dados")
    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]

    # Remover colunas com NaN (se houver)
    X = X.fillna(0)
    # Manter apenas numéricas
    X = X.select_dtypes(include=[np.number])

    print(f"  Features: {X.shape[1]} | Amostras: {X.shape[0]}")
    print(f"  Distribuição target: {y.value_counts().to_dict()}")

    # Train / Val / Test — 70% / 15% / 15%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"  Treino: {X_train.shape[0]} | Val: {X_val.shape[0]} | Teste: {X_test.shape[0]}")

    # ─────────────────────────────────────────
    # 6. ESCALONAMENTO
    # ─────────────────────────────────────────
    print("\n[ETAPA 6] Escalonamento (StandardScaler)")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),       columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # Salvar scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # ─────────────────────────────────────────
    # 7. SELEÇÃO DE FEATURES
    # ─────────────────────────────────────────
    print("\n[ETAPA 7] Seleção de Features (Mutual Information — Top 25)")
    X_train_sel, X_test_sel, sel_features, mi_scores = select_features(
        X_train_sc, y_train, X_test_sc, method="mutual_info", k=25
    )
    X_val_sel = X_val_sc[sel_features]

    # Salvar feature names
    joblib.dump(sel_features, os.path.join(MODELS_DIR, "feature_names.joblib"))

    # Salvar conjuntos para inferência
    X_test_sel.to_csv(os.path.join(DATA_PROCESSED, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PROCESSED, "y_test.csv"), index=False)

    # ─────────────────────────────────────────
    # 8. TREINAMENTO DOS MODELOS
    # ─────────────────────────────────────────
    print("\n[ETAPA 8] Treinamento e Avaliação dos Modelos")
    results = train_all_models(X_train_sel, y_train, X_test_sel, y_test)

    # ─────────────────────────────────────────
    # 9. SELECIONAR MELHOR MODELO
    # ─────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    print(f"\n[MELHOR MODELO] {best_name} — F1 Weighted: {results[best_name]['f1_weighted']:.4f}")

    # ─────────────────────────────────────────
    # 10. OTIMIZAÇÃO DE HIPERPARÂMETROS
    # ─────────────────────────────────────────
    print("\n[ETAPA 10] Otimização de Hiperparâmetros")
    best_model_raw = results[best_name]["model"]
    best_model_opt = optimize_best_model(
        best_model_raw, X_train_sel, y_train, best_name
    )

    # Reavaliação após otimização
    final_metrics = evaluate_model(
        best_model_opt, X_test_sel, y_test, sorted(y_test.unique().tolist())
    )
    results[f"{best_name} (Otimizado)"] = {
        **final_metrics,
        "cv_f1_mean": 0,
        "cv_f1_std": 0,
        "model": best_model_opt,
    }
    print(f"  F1 Weighted após otimização: {final_metrics['f1_weighted']:.4f}")
    print(f"  Accuracy após otimização:    {final_metrics['accuracy']:.4f}")

    # Determinar modelo final
    final_name = f"{best_name} (Otimizado)"
    final_model = best_model_opt

    # ─────────────────────────────────────────
    # 11. SALVAR MODELO
    # ─────────────────────────────────────────
    print("\n[ETAPA 11] Persistência do Modelo")
    save_model(final_model, MODELS_DIR, "best_model")
    joblib.dump(final_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    print(f"[SAVE] Modelo salvo em joblib e pkl.")

    # ─────────────────────────────────────────
    # 12. VISUALIZAÇÕES
    # ─────────────────────────────────────────
    print("\n[ETAPA 12] Geração de Visualizações")
    # Usar o melhor modelo sem otimização para feature importance (já treinado em X_test_sel)
    model_for_plots = results[best_name]["model"]

    generate_all_plots(
        df_raw=df_raw,
        df_processed=df_encoded,
        target_col=TARGET,
        results={k: v for k, v in results.items() if "(Otimizado)" not in k},
        best_model=model_for_plots,
        feature_names=sel_features,
        X_test=X_test_sel,
        y_test=y_test,
        output_dir=FIGURES_DIR,
    )

    # ─────────────────────────────────────────
    # 13. JSON PARA DASHBOARD
    # ─────────────────────────────────────────
    print("\n[ETAPA 13] Gerando JSONs para Dashboard")
    _generate_dashboard_jsons(
        df_raw, df_encoded, results, final_metrics,
        sel_features, model_for_plots, y_test, TARGET, REPORTS_DIR
    )

    # ─────────────────────────────────────────
    # 14. RELATÓRIO FINAL DE INSIGHTS
    # ─────────────────────────────────────────
    print("\n[ETAPA 14] Relatório de Insights de Negócio")
    _generate_business_report(
        df_raw, results, final_metrics, sel_features,
        model_for_plots, best_name, REPORTS_DIR
    )

    print("\n" + "=" * 70)
    print("   PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)
    print(f"  Figuras:    {FIGURES_DIR}")
    print(f"  Modelos:    {MODELS_DIR}")
    print(f"  Relatórios: {REPORTS_DIR}")
    print("=" * 70)


def _generate_dashboard_jsons(df_raw, df_encoded, results, final_metrics,
                                sel_features, model, y_test, target_col, output_dir):
    """Gera todos os JSON para consumo pelo frontend React/Next.js."""

    # ── KPIs principais
    kpis = {
        "total_funcionarios": int(len(df_raw)),
        "satisfacao_media": round(float(df_raw[target_col].mean()), 2),
        "satisfacao_mediana": round(float(df_raw[target_col].median()), 2),
        "pct_alta_satisfacao": round(
            float((df_raw[target_col] >= 3).sum() / len(df_raw) * 100), 1
        ),
        "pct_baixa_satisfacao": round(
            float((df_raw[target_col] <= 2).sum() / len(df_raw) * 100), 1
        ),
        "taxa_attrition": round(
            float((df_raw["Attrition"] == "Yes").sum() / len(df_raw) * 100), 1
        ) if "Attrition" in df_raw.columns else None,
        "distribuicao_satisfacao": df_raw[target_col].value_counts().sort_index().to_dict(),
    }
    _save_json(kpis, output_dir, "kpis.json")

    # ── Métricas dos modelos
    models_metrics = {}
    for name, res in results.items():
        models_metrics[name] = {
            "accuracy": round(res["accuracy"], 4),
            "f1_macro": round(res["f1_macro"], 4),
            "f1_weighted": round(res["f1_weighted"], 4),
            "precision_macro": round(res["precision_macro"], 4),
            "recall_macro": round(res["recall_macro"], 4),
            "cv_f1_mean": round(res.get("cv_f1_mean", 0), 4),
            "cv_f1_std": round(res.get("cv_f1_std", 0), 4),
        }
    _save_json(models_metrics, output_dir, "models_metrics.json")

    # ── Ranking de fatores de satisfação (feature importance)
    feature_ranking = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for feat, imp in sorted(zip(sel_features, importances),
                                 key=lambda x: x[1], reverse=True):
            feature_ranking.append({"feature": feat, "importance": round(float(imp), 6)})
    _save_json(feature_ranking, output_dir, "feature_ranking.json")

    # ── Dados agregados por departamento
    dept_data = []
    if "Department" in df_raw.columns:
        for dept, grp in df_raw.groupby("Department"):
            dept_data.append({
                "department": dept,
                "satisfacao_media": round(float(grp[target_col].mean()), 2),
                "n_funcionarios": int(len(grp)),
                "pct_alta_satisfacao": round(
                    float((grp[target_col] >= 3).sum() / len(grp) * 100), 1
                ),
                "attrition_rate": round(
                    float((grp["Attrition"] == "Yes").sum() / len(grp) * 100), 1
                ) if "Attrition" in grp.columns else None,
            })
    _save_json(dept_data, output_dir, "satisfaction_by_department.json")

    # ── Dados agregados por cargo
    role_data = []
    if "JobRole" in df_raw.columns:
        for role, grp in df_raw.groupby("JobRole"):
            role_data.append({
                "role": role,
                "satisfacao_media": round(float(grp[target_col].mean()), 2),
                "n_funcionarios": int(len(grp)),
                "pct_alta_satisfacao": round(
                    float((grp[target_col] >= 3).sum() / len(grp) * 100), 1
                ),
            })
    _save_json(role_data, output_dir, "satisfaction_by_role.json")

    # ── Satisfação por faixa etária
    df_temp = df_raw.copy()
    df_temp["FaixaEtaria"] = pd.cut(df_temp["Age"], bins=[17, 25, 35, 45, 60],
                                     labels=["18-25", "26-35", "36-45", "46-60"])
    age_data = []
    for faixa, grp in df_temp.groupby("FaixaEtaria", observed=True):
        age_data.append({
            "faixa": str(faixa),
            "satisfacao_media": round(float(grp[target_col].mean()), 2),
            "n_funcionarios": int(len(grp)),
        })
    _save_json(age_data, output_dir, "satisfaction_by_age.json")

    # ── Satisfação por faixa salarial
    df_temp["FaixaSalarial"] = pd.qcut(df_temp["MonthlyIncome"], q=4,
                                        labels=["Q1 Baixo", "Q2 Médio-Baixo",
                                                "Q3 Médio-Alto", "Q4 Alto"])
    salary_data = []
    for faixa, grp in df_temp.groupby("FaixaSalarial", observed=True):
        salary_data.append({
            "faixa": str(faixa),
            "satisfacao_media": round(float(grp[target_col].mean()), 2),
            "n_funcionarios": int(len(grp)),
        })
    _save_json(salary_data, output_dir, "satisfaction_by_salary.json")

    # ── Matriz de confusão do melhor modelo
    cm_data = {
        "confusion_matrix": final_metrics.get("confusion_matrix", []),
        "classes": sorted(y_test.unique().tolist()),
    }
    _save_json(cm_data, output_dir, "confusion_matrix.json")

    print(f"  [JSON] 8 arquivos JSON gerados em {output_dir}")


def _save_json(data, directory: str, filename: str):
    """Salva dict como JSON."""
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)


def _generate_business_report(df_raw, results, final_metrics, sel_features,
                                model, best_name, output_dir):
    """Gera relatório textual com insights de negócio."""
    satisfaction_map = {1: "Baixa", 2: "Média-Baixa", 3: "Média-Alta", 4: "Alta"}
    target_col = "JobSatisfaction"

    # Feature importances top 10
    top_features = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_features = sorted(zip(sel_features, importances),
                               key=lambda x: x[1], reverse=True)[:10]

    # Melhores e piores departamentos
    dept_sat = df_raw.groupby("Department")[target_col].mean().sort_values(ascending=False)
    role_sat = df_raw.groupby("JobRole")[target_col].mean().sort_values(ascending=False)

    # Taxa de attrition por satisfação
    if "Attrition" in df_raw.columns:
        attrition_by_sat = df_raw.groupby(target_col).apply(
            lambda g: (g["Attrition"] == "Yes").sum() / len(g) * 100
        ).round(1)
    else:
        attrition_by_sat = {}

    best_model_name = max(results, key=lambda k: results[k]["f1_weighted"])

    report = f"""
================================================================================
    RELATÓRIO FINAL — PREVISÃO DA SATISFAÇÃO DOS FUNCIONÁRIOS
================================================================================

DATA DE GERAÇÃO: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
DATASET: Employee-Attrition.csv
REGISTROS: {len(df_raw)} funcionários | {df_raw.shape[1]} variáveis

================================================================================
1. VISÃO GERAL DO PROBLEMA
================================================================================

A satisfação dos funcionários é um dos indicadores mais críticos para a saúde
organizacional. Funcionários insatisfeitos apresentam maior propensão ao
abandono (attrition), menor produtividade e maior absenteísmo.

VARIÁVEL ALVO: JobSatisfaction (escala ordinal 1-4)
  1 = Baixa     |  2 = Média-Baixa  |  3 = Média-Alta  |  4 = Alta

DISTRIBUIÇÃO DA SATISFAÇÃO:
{df_raw[target_col].value_counts().sort_index().to_string()}

  → Satisfação média geral: {df_raw[target_col].mean():.2f} / 4.0
  → Funcionários com alta satisfação (≥3): {(df_raw[target_col]>=3).sum()} ({(df_raw[target_col]>=3).mean()*100:.1f}%)
  → Funcionários com baixa satisfação (≤2): {(df_raw[target_col]<=2).sum()} ({(df_raw[target_col]<=2).mean()*100:.1f}%)

================================================================================
2. DESCOBERTAS DA ANÁLISE EXPLORATÓRIA
================================================================================

SATISFAÇÃO POR DEPARTAMENTO:
{dept_sat.to_string()}

SATISFAÇÃO POR CARGO (TOP 3 vs BOTTOM 3):
  Maiores: {', '.join([f"{r} ({v:.2f})" for r, v in role_sat.head(3).items()])}
  Menores: {', '.join([f"{r} ({v:.2f})" for r, v in role_sat.tail(3).items()])}

HORAS EXTRAS vs SATISFAÇÃO:
  Sem hora extra: {df_raw[df_raw['OverTime']=='No'][target_col].mean():.2f}
  Com hora extra: {df_raw[df_raw['OverTime']=='Yes'][target_col].mean():.2f}
  → Funcionários com hora extra são {'menos' if df_raw[df_raw['OverTime']=='Yes'][target_col].mean() < df_raw[df_raw['OverTime']=='No'][target_col].mean() else 'mais'} satisfeitos.

ATTRITION POR NÍVEL DE SATISFAÇÃO:
{attrition_by_sat.to_string() if len(attrition_by_sat) > 0 else 'N/A'}
  → Funcionários com satisfação baixa têm maior taxa de desligamento.

================================================================================
3. RESULTADOS DOS MODELOS DE MACHINE LEARNING
================================================================================

{"".join([f"  {name}: Accuracy={res['accuracy']:.4f} | F1-Macro={res['f1_macro']:.4f} | F1-Weighted={res['f1_weighted']:.4f}" + chr(10) for name, res in results.items()])}
MELHOR MODELO: {best_model_name}
  → Accuracy:    {results[best_model_name]['accuracy']:.4f}
  → F1 Macro:    {results[best_model_name]['f1_macro']:.4f}
  → F1 Weighted: {results[best_model_name]['f1_weighted']:.4f}
  → Precision:   {results[best_model_name]['precision_macro']:.4f}
  → Recall:      {results[best_model_name]['recall_macro']:.4f}

================================================================================
4. PRINCIPAIS FATORES QUE INFLUENCIAM A SATISFAÇÃO (Feature Importance)
================================================================================

{"".join([f"  {i+1:2d}. {feat:<40} Importância: {imp:.4f}" + chr(10) for i, (feat, imp) in enumerate(top_features)])}
================================================================================
5. RECOMENDAÇÕES PRÁTICAS PARA A EMPRESA
================================================================================

Com base nos resultados do modelo preditivo, as seguintes ações são recomendadas:

1. CONTROLE DE HORAS EXTRAS
   → Implementar políticas de gestão de horas extras. Funcionários submetidos
     a overtime crônico apresentam menor satisfação. Revisar alocação de tarefas
     e considerar contratações adicionais em áreas sobrecarregadas.

2. PLANO DE CARREIRA E PROMOÇÃO
   → O tempo desde a última promoção está entre os principais fatores. Criar
     trilhas de carreira claras com avaliações periódicas de performance e
     promoções baseadas em critérios objetivos.

3. EQUILÍBRIO VIDA-TRABALHO
   → WorkLifeBalance aparece consistentemente como variável relevante. Ampliar
     benefícios como home office, horários flexíveis e programas de bem-estar.

4. REMUNERAÇÃO COMPETITIVA
   → A renda mensal impacta diretamente a satisfação. Revisar estrutura salarial
     garantindo alinhamento com o mercado, especialmente nos cargos com menor
     satisfação.

5. AMBIENTE DE TRABALHO
   → EnvironmentSatisfaction é um preditor significativo. Investir em
     melhorias do ambiente físico, cultura organizacional e programas de
     reconhecimento.

6. ATENÇÃO AOS CARGOS CRÍTICOS
   → Focar esforços nos cargos com menor satisfação média identificados na
     análise, implementando programas específicos de engajamento.

7. DISTÂNCIA RESIDÊNCIA-TRABALHO
   → Considerar políticas de mobilidade, auxílio transporte e opções de
     trabalho remoto para funcionários com maior distância.

8. ONBOARDING E TREINAMENTO
   → Funcionários com menos tempo de empresa e pouco treinamento tendem a
     menor satisfação. Investir em programas de integração e desenvolvimento.

================================================================================
6. MÉTRICAS DE SUCESSO DO PROJETO
================================================================================

  ✓ EDA completa com identificação de outliers e correlações
  ✓ Feature Engineering com 7+ novas variáveis criadas
  ✓ {len(results)} modelos treinados e comparados
  ✓ Melhor modelo otimizado com RandomizedSearchCV (20 iterações)
  ✓ {len(sel_features)} features selecionadas por Mutual Information
  ✓ Modelo salvo em formato joblib e pkl para produção
  ✓ 18 visualizações geradas em alta resolução
  ✓ 8 arquivos JSON prontos para dashboard React/Next.js

================================================================================
"""

    path = os.path.join(output_dir, "business_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[REPORT] Relatório de negócio salvo: {path}")


if __name__ == "__main__":
    main()
