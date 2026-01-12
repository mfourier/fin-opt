# FinOpt: Marco Matematico y Arquitectura Convexa

> Documento tecnico redactado en estilo academico (doctorado en optimizacion convexa). Se explican las formulaciones, supuestos y conexiones entre modulos.

## 1. Vision General
- Objetivo: encontrar el horizonte minimo $T^*$ y la politica de asignacion $X^*$ que satisfacen metas probabilisticas de riqueza bajo ingresos y retornos estocasticos.
- Enfoque: simulacion Monte Carlo + reformulacion CVaR de restricciones de chance + busqueda sobre $T$ (bilevel).
- Propiedad clave: la riqueza $W_t^m(X)$ es afina en $X$, lo que mantiene convexidad tras la reformulacion CVaR.

## 2. Arquitectura de Modulos
- `income.py`: genera flujos de ingreso (fijo + variable) y contribuciones $A_t$.
- `returns.py`: genera retornos lognormales correlacionados $R_t^m$.
- `portfolio.py`: ejecuta dinamica de riqueza y calcula factores de acumulacion $F_{s,t}^m$.
- `goals.py`: define metas intermedias y terminales como restricciones de probabilidad.
- `optimization.py`: resuelve el problema convexo interno (CVaROptimizer) y realiza la busqueda externa de horizonte (GoalSeeker).
- `model.py`: orquesta todo (simulacion, cacheo, ploteo, optimizacion end-to-end).

## 3. Modelado de Ingresos (`income.py`)
### 3.1 Ingreso fijo
- Serie determinista: $y_t^{\text{fixed}} = \text{base} \,(1+m)^t$ con $m = (1+g)^{1/12}-1$.
- Aumentos salariales en fechas especificas se convierten a offsets mensuales antes de compounding.

### 3.2 Ingreso variable
- Tendencia con crecimiento anual $g$: factor mensual $m$.
- Estacionalidad $s_{(t \bmod 12)}$ multiplicativa.
- Ruido gaussiano $\epsilon_t \sim \mathcal{N}(0,\sigma^2)$ aplicado sobre la media $\mu_t$.
- Guardas: floor/cap y truncamiento a no-negatividad.

Proyeccion estocastica:
$$
\tilde{Y}_t = \mu_t (1+\epsilon_t), \quad \mu_t = \text{base}\,(1+m)^t s_{(t\bmod12)}
$$
$$
Y_t^{\text{var}} = \min(\text{cap},\,\max(\text{floor},\,\tilde{Y}_t))
$$

### 3.3 Contribuciones
- Esquema rotatorio de 12 meses con factores $\alpha^f,\alpha^v \in [0,1]^{12}$.
- Formula: $A_t = \alpha^f_{(t+o)\bmod12}\, y_t^{\text{fixed}} + \alpha^v_{(t+o)\bmod12}\, Y_t^{\text{var}}$, donde $o$ depende de la fecha de inicio.
- Salida vectorizada: $(n_{\text{sims}}, T)$.

## 4. Generacion de Retornos (`returns.py`)
- Modelo: $1+R_t^m \sim \text{LogNormal}(\mu_{\log}^m,\Sigma)$.
- Conversion aritmetica a log-espacio:
$$
\sigma_{\log} = \sqrt{\log\!\left(1 + \frac{\sigma_{\text{arith}}^2}{(1+\mu_{\text{arith}})^2}\right)},\qquad
\mu_{\log} = \log(1+\mu_{\text{arith}}) - \tfrac{1}{2}\sigma_{\log}^2
$$
- Covarianza log-espacio: $\Sigma = D\,\rho\,D$, $D=\operatorname{diag}(\sigma_{\log})$; $\rho$ validado simetrico PSD.
- Muestras: $Z\sim\mathcal{N}(\mu_{\log},\Sigma)$, $R=\exp(Z)-1$ (garantiza $R>-1$).

## 5. Dinamica de Riqueza y Portafolio (`portfolio.py`)
### 5.1 Formulacion recursiva
$$
W_{t+1}^m = (W_t^m + A_t x_t^m)(1+R_t^m)
$$

### 5.2 Representacion afina (cerrada)
$$
\boxed{W_t^m(X) = W_0^m F_{0,t}^m + \sum_{s=0}^{t-1} A_s x_s^m F_{s,t}^m}
$$
con $F_{s,t}^m = \prod_{r=s}^{t-1}(1+R_r^m)$.

Consecuencias:
- Afinidad en $X$: convexidad preservada al insertar en restricciones CVaR.
- Gradiente analitico: $\partial W_t^m / \partial x_s^m = A_s F_{s,t}^m$.
- Complejidad memoria de $F$: $O(n_{\text{sims}} T^2 M)$.

### 5.3 Simplex de asignacion
$$
\mathcal{X}_T = \{X\in\mathbb{R}^{T\times M}: x_t^m\ge0, \sum_m x_t^m=1\ \forall t\}
$$
Producto cartesiano de $T$ simplices de dimension $M-1$.

## 6. Metas como Restricciones de Probabilidad (`goals.py`)
- Intermedia: $\mathbb{P}(W_{t_g}^{m_g}\ge b_g) \ge 1-\varepsilon_g$ con $t_g$ fijo (mes o fecha -> offset 1-indexado).
- Terminal: $\mathbb{P}(W_T^{m_g}\ge b_g) \ge 1-\varepsilon_g$ evaluada en el horizonte variable $T$.
- `GoalSet` valida duplicados, resuelve cuentas, calcula $T_{\min} = \max t_g$ y cachea offsets.

## 7. Reformulacion CVaR (`optimization.py`)
### 7.1 De chance constraint a CVaR
Restriccion original: $\mathbb{P}(W \ge b) \ge 1-\varepsilon$.
Reemplazo conservador:
$$
\text{CVaR}_{\varepsilon}(b-W) \le 0
$$

### 7.2 Forma epigrafica (LP/SOCP)
Variables: $\gamma \in \mathbb{R}$, $z \in \mathbb{R}_+^N$ para $N$ escenarios.
$$
\text{CVaR}_{\varepsilon}(L) = \min_{\gamma,z} \left\{ \gamma + \frac{1}{\varepsilon N}\sum_{i=1}^N z_i \right\}
$$
sujeto a $z_i \ge L_i - \gamma$, $z_i \ge 0$, $L_i = b - W^i$.

### 7.3 Ensamblaje de restricciones por meta
Para cada meta $g$ con umbral $b_g$, cuenta $m_g$ y tiempo $t_g$:
1) Construir $W_{t_g}^{m_g}(X)$ via forma afina.
2) Shortfall $L^i = b_g - W_{t_g}^{m_g,i}$.
3) Agregar variables $(\gamma_g, z_g)$ y restricciones CVaR.

### 7.4 Objetivos convexos soportados
1. `risky`: maximiza $\mathbb{E}[\sum_m W_T^m]$ (LP).
2. `balanced`: minimiza turnover cuadratico $-\sum_{t,m}(x_{t+1,m}-x_t^m)^2$ (convexo).
3. `risky_turnover`: riqueza esperada menos penalizacion L2 en rebalanceo.
4. `conservative`: $\mathbb{E}[W_T] - \lambda\,\text{Std}(W_T)$ (max de concava, mantiene DCP).

### 7.5 Programa convexo
- Decision: $X\in\mathcal{X}_T$, $(\gamma_g,z_g)$ por meta.
- Restricciones: simplex + CVaR por meta.
- Objetivo: depende de configuracion anterior.
- Solvers: ECOS (default), SCS, CLARABEL; soporta `max_iters` y `verbose`.
- Post-procesado: proyeccion a simplex si el solver entrega tolerancias numericas fuera de 1e-6.

## 8. Optimizacion Bilevel (GoalSeeker)
### 8.1 Planteamiento
Problema externo: $\min_{T\in\mathbb{N}} T$ sujeto a factibilidad de metas.
Problema interno: para $T$ fijo, resolver el programa convexo anterior.

### 8.2 Estrategias de busqueda
- Lineal: $T = T_{\text{start}}, \dots, T_{\max}$ (seguro, mas lento).
- Binaria: requiere intervalos factibles/infeasibles; acelera cuando metas son exigentes.
- Warm-start: extiende $X^*$ del horizonte previo para inicializar el siguiente ($X_{t=T} = X_{T-1}$).

### 8.3 Criterios de factibilidad
- Verificacion exacta con SAA (indicatriz): se simula con $X^*$ y se computa tasa de violacion empirica vs $\varepsilon$ por meta.
- Normalizacion de filas de $X$ si $\max|\sum_m x_t^m-1|>10^{-2}$ descarta la solucion.

## 9. Orquestacion (`model.py`)
- `FinancialModel.simulate`: genera $A$, $R$, ejecuta `Portfolio.simulate`, empaqueta `SimulationResult`.
- Semillas: ingresos usan `seed`, retornos `seed+1` para independencia; cacheo por hash de parametros.
- `FinancialModel.optimize`: instancia `GoalSeeker` + `CVaROptimizer`, realiza busqueda de horizonte y retorna `OptimizationResult`.
- `SimulationResult`: metricas (CAGR, Sharpe, Sortino, drawdown), VaR/CVaR terminal, resumenes y plots.

## 10. Consideraciones Numericas
- Validacion de correlacion: simetria, diagonal 1, autovalores >= -1e-10.
- Tolerancias solver: `max_iters` configurable; proyeccion simplex posterior.
- Escalamiento: memoria de $F$ domina para T grande; usar n_sims moderado o truncar horizonte en pruebas.
- Semillas separadas garantizan independencia entre procesos de ingreso y retorno.

## 11. Flujo de Datos End-to-End
1) `IncomeModel.contributions` produce $A$ (n_sims, T).
2) `ReturnModel.generate` produce $R$ (n_sims, T, M).
3) `Portfolio.simulate` computa $W$ y $F$ (afine/recursivo).
4) `CVaROptimizer.solve` construye restricciones CVaR usando $F$ y $A$.
5) `GoalSeeker` busca $T^*$; `OptimizationResult` almacena $X^*$, factibilidad y diagnosticos.
6) `SimulationResult` verifica metas in/out-of-sample y produce metricas.

## 12. Interpretacion y Alcance
- La reforma CVaR es implicativa: $\text{CVaR}_{\varepsilon}(b-W) \le 0 \Rightarrow \mathbb{P}(W\ge b)\ge 1-\varepsilon$, pero no viceversa; otorga holgura conservadora.
- Afinidad de $W(X)$ habilita globalidad via programacion convexa (evita minimos locales de SAA suave).
- Correlaciones lognormales aseguran $R>-1$, evitando colapsos imposibles en Monte Carlo.

## 13. Pistas para Extensiones
- Sustituir lognormal por distribuciones con colas gruesas (t-student) preservando PSD por transformaciones de varianza.
- Incorporar costos de transaccion mediante penalizacion L1 en rebalanceo (convexo) o restricciones por banda.
- Reemplazar CVaR por espectros de riesgo coherente (p.ej., Expected Shortfall ponderado) manteniendo epigrafica convexa.
- Parallel tempering de $T$ con evaluaciones en batch para reducir latencia en horizontes largos.

---
Documento alineado con el codigo actual en `src/` y las descripciones de `docs/`. Todo el contenido se mantiene en ASCII para compatibilidad.