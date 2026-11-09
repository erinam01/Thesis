# Literature review: online PCA with forgetting, sliding windows and direct updates

This review collects the literature behind the methods implemented in this project. For each topic it states what the cited works establish and where the result is used: `methods.py`, `notebooks/01_baselines.ipynb` (covariance and SVD-update methods), `notebooks/02_sliding_window.ipynb` (sliding windows), `notebooks/03_oja.ipynb` (Oja's method), `notebooks/04_sensitivity_analysis.ipynb` (robustness of the conclusions, fixed-gain sweep, spectral gap) and `notebooks/05_computational_cost.ipynb` (runtime of all methods in the same run). Section 7 lists what has not yet been checked against the original texts.

All methods estimate the leading principal direction of a zero-mean stream $x_1, x_2, \dots \in \mathbb R^d$ ($d=2$ in the experiments). They differ in how past observations are weighted and in what is updated: a covariance matrix, a truncated SVD, or the direction itself.

---

## 1. Online PCA and the memory–noise trade-off

Online PCA updates the estimate one observation at a time, without storing the stream. Cardot and Degras (2018) survey the main families of algorithms: perturbation methods, incremental methods (including SVD updating), and stochastic optimisation (Oja and Karhunen (1985), Sanger's generalized Hebbian algorithm). They compare them on accuracy, computation time and memory, and discuss the choice of tuning parameters. Their survey is reference [2] of the thesis proposal.

When the data are non-stationary, every recursive estimator faces the same trade-off. A long memory averages out sampling noise but reacts slowly to changes, and a short memory does the opposite. In system identification this trade-off is the subject of the survey by Ljung and Gunnarsson (1990). It studies how algorithms for time-varying systems, including recursive least squares with exponential forgetting, should be designed under different assumptions about how the true system varies.

## 2. Covariance-based methods

**Cumulative covariance** uses $C_n = n^{-1}\sum_{i\le n} x_i x_i^\top$, updated recursively, followed by an eigendecomposition. Every observation has the same weight, so the estimate stops adapting as $n$ grows (`cumulative_pca`, notebook 01, Section 1).

**Exponential forgetting** replaces $1/n$ by a fixed gain: $C_n=(1-\alpha)C_{n-1}+\alpha x_nx_n^\top$, so an observation of age $k$ has weight $\alpha(1-\alpha)^k$ (eq. 1 of the proposal; `forgetting_pca`, notebook 01, Section 2). This is the covariance analogue of the exponentially weighted cost of recursive least squares, where the forgetting factor $\lambda=1-\alpha$ is the classical tool for tracking (Ljung and Gunnarsson, 1990). In process monitoring, Li et al. (2000) use recursive PCA with rank-one modifications of the correlation matrix for adaptive monitoring.

## 3. SVD-update methods

### 3.1 Rank-one modifications

All SVD-update methods in this project rely on the same identity. If $S\approx U\,\mathrm{diag}(s^2)\,U^\top$ and a sample $x$ is added or removed, then

$$S\pm xx^\top=[U\ \hat\rho]\Big(\begin{bmatrix}\mathrm{diag}(s^2)&0\\0&0\end{bmatrix}\pm zz^\top\Big)[U\ \hat\rho]^\top,\qquad z=\begin{bmatrix}U^\top x\\ \lVert\rho\rVert\end{bmatrix},\quad \rho=x-UU^\top x,$$

so only a small $(r+1)\times(r+1)$ eigenproblem has to be solved. Golub (1973) studies the eigenvalues of a matrix modified by a rank-one term. Bunch, Nielsen and Sorensen (1978) study this small problem, a diagonal matrix plus a rank-one term, and give an algorithm for its eigenvalues and eigenvectors (the project solves it directly with a standard symmetric eigensolver, since in two dimensions it is at most $3\times3$). Brand (2006) gives a unified treatment of updates, downdates and other low-rank modifications of a thin SVD for streaming data. In the project, `svd_rank_one_modify` implements one update ($+$) or downdate ($-$) step.

### 3.2 Incremental PCA (all observations, truncated SVD)

Arora, Cotter, Livescu and Srebro (2012) state the incremental algorithm in exactly this form for the second-moment matrix: $C^{(t)}=P_{\text{rank-}k}\big(C^{(t-1)}+x_tx_t^\top\big)$, computed through the small eigenproblem above, with cost $O(k^2d)$ per step. They also give an example of orthogonal data for which the truncated algorithm converges to the wrong direction. Their example shows that truncation can change the result, not only its accuracy. This is the method of `incremental_pca` (rank 1, $\alpha_n=1/n$; notebook 01, Section 3).

Related incremental eigenspace methods are the sequential Karhunen–Loève algorithm of Levy and Lindenbaum (2000) and the eigenspace merging of Hall, Marshall and Martin (2000), which also updates the mean. `sklearn.decomposition.IncrementalPCA` implements the algorithm of Ross et al. (2008), an extension of Levy and Lindenbaum's method. According to its documentation it centres the data with the running sample mean and has no forgetting factor, which is why it is not used here (notebook 01, Section 3).

### 3.3 SVD update with forgetting

Levy and Lindenbaum (2000) propose a forgetting factor in the sequential Karhunen–Loève update. Ross et al. (2008, §3.1.1) analyse it: multiplying the retained singular values by $f\in[0,1]$ before each update reduces the contribution of each block of data to the covariance by a factor $f^2$ per update (their Lemma 2). Since the covariance is built from the squared singular values, the gain of eq. (1) corresponds to $f^2=1-\alpha$, i.e. $f=\sqrt{1-\alpha}$. They also define the effective size of the observation history through $n\leftarrow fn+m$, which uses $f$ and not $f^2$. With a fixed gain, `svd_pca` and `rank1_svd_pca` are the one-sample version of this method (notebook 01, Section 4).

Truncation and forgetting interact. A truncated update discards the perpendicular part of each new sample. With forgetting, the error introduced at one step is multiplied by $(1-\alpha)$ at every later step, so it does not accumulate. Without forgetting (Arora et al.'s example) or with a hard window (Section 4.2), it does.

Not accumulating does not mean that truncation has no effect. The truncation is applied at every step, so the rank-1 update is a different estimator from the full one. In 2D the full-rank SVD update with forgetting is exactly the covariance update of eq. (1); the only difference between the "SVD-based" and the "covariance-based" method is therefore the truncation. With the same gain, the rank-1 update is slightly less noisy in stationary data and slower after an abrupt change (notebook 01, Section 4): it is a different point of the noise–speed trade-off. A further consequence is that the rank-1 update does not estimate $\lambda_2$, so the spectral gap needed by the observable state of the proposal (eq. 12) must be obtained separately, for example from the trace updated with the same gain.

## 4. Sliding windows

### 4.1 Moving-window covariance and downdating

A sliding window keeps the last $w$ samples with equal weight. Updating its covariance needs one update and one downdate per step, $S_n=S_{n-1}+x_nx_n^\top-x_{n-w}x_{n-w}^\top$ (`sliding_window_cov_pca`; notebook 02, Section 1).

A downdate is a subtraction, so rounding errors accumulate and can be amplified. Chan, Golub and LeVeque (1983) analyse updating algorithms for the sample variance and the loss of accuracy caused by cancellation. Liang et al. (2008) give fast moving-window algorithms for QR and Cholesky factors. In the project, the rounding drift is negligible on Gaussian data, but a single very large observation leaves a permanent error after it leaves the window, which a periodic rebuild (`recompute_every`) removes (notebook 02, Section 4).

### 4.2 SVD update and downdate

The rank-one identity of Section 3.1 with a minus sign gives the downdate. In Brand (2006, Table 1 and eq. 10) a downdate removes a column of the data matrix and uses the corresponding row of the right singular vectors $V$, that is, how that column was represented; the covariance-form downdate of this project keeps no $V$ and removes $x_{n-w}x_{n-w}^\top$ in the current basis. Hall, Marshall and Martin (2000) remove a whole set of observations from an eigenspace model ("splitting"). They note that information discarded when the model was built cannot be recovered when observations are removed, and they advise caution when splitting (their Sections 4 and 7). With full rank (two components in 2D) update and downdate are exact. With rank 1 the downdate removes $x_{n-w}x_{n-w}^\top$ in full, although part of that sample was discarded when it entered (`sliding_window_svd_pca`; notebook 02, Sections 2 and 5). Yin et al. (2024, Section 3) name the same difficulty for sketches over sliding windows: after a rank-limited update the basis rotates, so earlier summaries are no longer aligned with the current singular vectors, and the error analysis has to account for it.

### 4.3 Subspace tracking on a window

The signal-processing literature tracks the dominant subspace directly, without forming the covariance. Badeau, David and Richard (2005) introduce the approximated power iteration (API) and its fast versions, and show that earlier trackers of this family are approximations of it (their Section VII). Their algorithm is written for exponential windows and for a truncated window (their Section II)

$$C(t)=\sum_{u=t-l+1}^{t}\beta^{t-u}x(u)x(u)^\top,\qquad C(t)=\beta C(t-1)+x(t)x(t)^\top-\beta^l x(t-l)x(t-l)^\top,$$

which contains both the rectangular window ($\beta=1$) and exponential forgetting ($l\to\infty$). The project uses only the rectangular window ($\beta=1$), although `sliding_window_cov_pca` and `tw_api_pca` also accept $\beta<1$.

Their truncated-window algorithm (TW-API) keeps, for each stored sample, its compressed coordinate in the current basis and rotates it with the basis. The leaving sample is therefore removed consistently with what was added. `tw_api_pca` implements TW-API for one direction. In the experiments it removes the persistent error of the truncated rank-1 downdate (notebook 02, Sections 3 and 5).

### 4.4 Windows with less memory

An exact window stores $w$ samples. When this is too costly, streaming algorithms keep approximate summaries over sliding windows instead (exponential histograms, sampling, Frequent Directions sketches); Yin et al. (2024, Section 1 and Table 1) review these approaches. However, note that in 2D the buffer is negligible.

## 5. Direct update of the principal direction: Oja's method

### 5.1 The algorithm

Oja (1982) shows that a linear neuron with a normalized Hebbian update extracts the principal component of a stationary input. Oja and Karhunen (1985) study the stochastic-approximation version for the leading eigenvectors of the expectation of a random matrix. The thesis proposal uses the normalized form with a controlled gain (eq. 2–3):

$$\tilde q_{n+1}=q_n+\alpha_n(I-q_nq_n^\top)x_{n+1}x_{n+1}^\top q_n,\qquad q_{n+1}=\tilde q_{n+1}/\lVert\tilde q_{n+1}\rVert$$

(`oja_pca`; notebook 03, Section 1). Related direct methods are the other stochastic-gradient algorithms surveyed by Cardot and Degras (2018, Section 5), such as Sanger's generalized Hebbian algorithm, and CCIPCA (Weng, Zhang and Hwang, 2003), which estimates components without a covariance matrix (Cardot and Degras, 2018, Section 5.3).

### 5.2 Convergence and the role of the spectral gap

The continuous-time flow of Oja's rule converges exponentially, and from almost every initial condition to the principal eigenspace (Yan, Helmke and Moore, 1994). Oja and Karhunen (1985, Lemma 2, eqs. 12–13) give the mean dynamics of the stochastic update: near the leading direction, the component along the $i$-th eigenvector decays at rate $\lambda_1-\lambda_i$ per unit of gain. The spectral gap therefore acts as a scale for the gain of Oja's method; notebook 03, Section 2, uses it to choose the Oja gains.

For the stochastic algorithm, finite-sample guarantees with decreasing steps are given by Balsubramani, Dasgupta and Freund (2013) and Jain et al. (2016). In both, the rate depends on the eigengap $\lambda_1-\lambda_2$. Jain et al. (2016, Theorems 1.2–1.3) use a step $\eta_i\propto1/\big((\lambda_1-\lambda_2)(\beta+i)\big)$, i.e. divided by the gap, and note that how to set the step size "in a robust and data driven manner" is an important open question.

For a **constant** step $\beta$, Li, Wang, Liu and Zhang (2018, Theorem 1, eq. 2.7) give an **upper bound** on the error of Oja's iteration $u^{(n)}=\Pi\{u^{(n-1)}+\beta X^{(n)}X^{(n)\top}u^{(n-1)}\}$: a transient that contracts like $(1-\beta(\lambda_1-\lambda_2))^{2n}$ (Lemma 10) plus a stationary term proportional to $\beta\sum_{k\ge2}\lambda_1\lambda_k/(\lambda_1-\lambda_k)$, with an unspecified constant. Their iteration has no projection $(I-qq^\top)$, but it coincides with eq. (2)–(3) to first order in the step. Notebook 03 uses only the dependence of this bound on the gap, not its constant.

## 6. Comparing methods at equal memory

Comparing a window with forgetting, or Oja with the covariance method, requires a common notion of memory. Two conventions appear in the literature.

- **Time constant.** The exponential weights decay by $e$ after $1/(1-\lambda)$ steps. Badeau et al. (2005) compare a sliding window of length $l=120$ with forgetting $\beta\approx0.99$, chosen so that the effective window lengths are equal. Their equality is $l\approx1/(1-\beta)$, and they conclude that sliding windows track abrupt changes faster. Ross et al. (2008) use the same form for the effective size of the history, $n=m/(1-f)$, but written in terms of their factor $f$ on the singular values. With $f=\sqrt{1-\alpha}\approx1-\alpha/2$ (Section 3.3) this is about $2m/\alpha$, i.e. the equal-noise size below rather than the time constant $1/\alpha$.
- **Equal noise.** For independent observations with common variance $\sigma^2$, the weighted average $\sum_k c_kx_k/\sum_k c_k$ has variance $\sigma^2\sum_k c_k^2/(\sum_k c_k)^2$, the same as a plain mean of $(\sum_k c_k)^2/\sum_k c_k^2$ observations; this number is the effective sample size of Kish (1965). For exponential weights this is $(2-\alpha)/\alpha\approx 2/\alpha$, twice the time constant. A window of length $w=2/\alpha-1$ therefore has the same stationary noise as forgetting with gain $\alpha$. With this convention forgetting is the faster method after an abrupt change (notebook 02, Section 7), with the default recovery threshold; with a threshold of $0.01$ the window recovers first, while the post-change error favours forgetting in every case (notebook 04, A1–A2). The two findings are compatible: they come from different matchings.

For the angle of the leading eigenvector, Anderson (1963) gives the asymptotic variance of the sample eigenvector, $\lambda_1\lambda_2/(n(\lambda_1-\lambda_2)^2)$ in 2D. With $n$ replaced by the effective sample size $2/\alpha$, this gives the stationary error of forgetting PCA, $\alpha\lambda_1\lambda_2/(2(\lambda_1-\lambda_2)^2)$.

For Oja's method the project uses the gain $\alpha_{\text{Oja}}=\alpha_{\text{cov}}/(\lambda_1-\lambda_2)$. This rule is our implementation: the scale by the gap is motivated by the mean dynamics of Oja and Karhunen (1985) and is consistent with the step sizes of Jain et al. (2016) and the bound of Li et al. (2018) (Section 5.2), but we have not found the rule stated in this form in the literature. It is checked numerically: in notebook 03, Section 2, the stationary errors of the two methods agree within $7\%$ at gap $4.5$, and in notebook 04, Part C, within $10\%$ down to gap $2.5$ for gains up to $\alpha=0.1$, less accurately at gap $1.5$. A derivation of the rule is left to the thesis text.


## References

- T. W. Anderson (1963). Asymptotic theory for principal component analysis. *Annals of Mathematical Statistics*, 34, 122–148. doi:10.1214/aoms/1177704248
- R. Arora, A. Cotter, K. Livescu, N. Srebro (2012). Stochastic optimization for PCA and PLS. *50th Allerton Conference on Communication, Control, and Computing*, 861–868. doi:10.1109/Allerton.2012.6483308
- R. Badeau, B. David, G. Richard (2005). Fast approximated power iteration subspace tracking. *IEEE Transactions on Signal Processing*, 53(8), 2931–2941. doi:10.1109/TSP.2005.850378
- A. Balsubramani, S. Dasgupta, Y. Freund (2013). The fast convergence of incremental PCA. *Advances in Neural Information Processing Systems 26*, 3174–3182. 10.5555/2999792.2999966
- M. Brand (2006). Fast low-rank modifications of the thin singular value decomposition. *Linear Algebra and its Applications*, 415(1), 20–30. doi:10.1016/j.laa.2005.07.021 
- J. R. Bunch, C. P. Nielsen, D. C. Sorensen (1978). Rank-one modification of the symmetric eigenproblem [Abstract]. *Numerische Mathematik*, 31(1), 31–48. doi:10.1007/BF01396012
- H. Cardot, D. Degras (2018). Online principal component analysis in high dimension: which algorithm to choose? *International Statistical Review*, 86(1), 29–50. doi:10.1111/insr.12220
- T. F. Chan, G. H. Golub, R. J. LeVeque (1983). Algorithms for computing the sample variance: analysis and recommendations. *The American Statistician*, 37(3), 242–247. doi:10.1080/00031305.1983.10483115 (content checked on the technical-report version: Yale University, Department of Computer Science, Technical Report #222)
- G. H. Golub (1973). Some modified matrix eigenvalue problems. *SIAM Review*, 15(2), 318–334. doi:10.1137/1015032
- P. Hall, D. Marshall, R. Martin (2000). Merging and splitting eigenspace models. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(9), 1042–1049. doi:10.1109/34.877525
- P. Jain, C. Jin, S. Kakade, P. Netrapalli, A. Sidford (2016). Streaming PCA: matching matrix Bernstein and near-optimal finite sample guarantees for Oja's algorithm. arXiv:1602.06929v2. (Theorem numbers and the quotation refer to this arXiv version. The paper also appeared in *Conference on Learning Theory (COLT)*, PMLR 49, 1147–1164.)
- L. Kish (1965). *Survey Sampling*. Wiley, New York.
- A. Levy, M. Lindenbaum (2000). Sequential Karhunen–Loève basis extraction and its application to images. *IEEE Transactions on Image Processing*, 9(8), 1371–1374. doi:10.1109/83.855432
- C. J. Li, M. Wang, H. Liu, T. Zhang (2018). Near-optimal stochastic approximation for online principal component estimation. *Mathematical Programming*, 167(1), 75–97. doi:10.1007/s10107-017-1182-z
- W. Li, H. H. Yue, S. Valle-Cervantes, S. J. Qin (2000). Recursive PCA for adaptive process monitoring. *Journal of Process Control*, 10(5), 471–486. doi:10.1016/S0959-1524(00)00022-6
- W. Liang, U. Kruger, X. Wang, L. Xie, T. Littler (2008). Fast moving window algorithm for QR and Cholesky decompositions. *Proc. 17th IFAC World Congress*, Seoul (*IFAC Proceedings Volumes*, 41(2)), 10106–10111. doi:10.3182/20080706-5-KR-1001.01710
- L. Ljung, S. Gunnarsson (1990). Adaptation and tracking in system identification—A survey. *Automatica*, 26(1), 7–21. doi:10.1016/0005-1098(90)90154-A
- E. Oja (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15, 267–273. doi:10.1007/BF00275687
- E. Oja, J. Karhunen (1985). On stochastic approximation of the eigenvectors and eigenvalues of the expectation of a random matrix. *Journal of Mathematical Analysis and Applications*, 106(1), 69–84. doi:10.1016/0022-247X(85)90131-3
- D. A. Ross, J. Lim, R.-S. Lin, M.-H. Yang (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125–141. doi:10.1007/s11263-007-0075-7
- J. Weng, Y. Zhang, W.-S. Hwang (2003). A fast algorithm for incremental principal component analysis. *Intelligent Data Engineering and Automated Learning (IDEAL 2003)*, Lecture Notes in Computer Science 2690, 876–881. doi:10.1007/978-3-540-45080-1_122
- W.-Y. Yan, U. Helmke, J. B. Moore (1994). Global analysis of Oja's flow for neural networks. *IEEE Transactions on Neural Networks*, 5(5), 674–683. doi:10.1109/72.317720
- H. Yin, D. Wen, J. Li, Z. Wei, X. Zhang, Z. Huang, F. Li (2024). Optimal matrix sketching over sliding windows. *Proceedings of the VLDB Endowment*, 17(9), 2149–2161. doi:10.14778/3665844.3665847
