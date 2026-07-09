# SSMtool (Haller-Group, ETH Zürich) — Vollständige Workflow-Dokumentation

Quelle: `/home/hz/Data/Attractor/SSMtool/` (geklont von `haller-group/SSMtool`)
Manual: `/home/hz/Data/Attractor/SSMtool/SSMtool_manual.pdf` (Ponsioen & Haller, 1. März 2019, 15 Seiten)
Stand der Software: SSMtool V1.0 (2017), Addendum_Isolas (17. Dezember 2018)

Diese Datei beschreibt die Arbeitsweise von SSMtool maximal granular: jeden Schritt der Pipeline, jede Funktion, jede Datei, jede mathematische Bedingung, mit wörtlichen Code-Kommentaren als Belege. Sie ist als Checkliste gedacht, anhand derer Codex (zweite KI) die Vollständigkeit gegen Quelltext und Manual abgleichen kann.

---

## 0. Inhaltsverzeichnis

1. Was ist SSMtool und was ist eine Spectral Submanifold (SSM)
2. Mathematischer Rahmen (Cabré-Fontich-de la Llave, Parametrization Method, Cohomologische Gleichungen)
3. Inputs, die SSMtool zwingend erwartet
4. Pipeline Schritt für Schritt mit Funktion + Code-Zitaten
5. Datei-für-Datei-Inventar (`SSMtool/SSMtool/` und `Addendum_Isolas/`)
6. Demos und Beispiele
7. Outputs
8. Limitationen
9. Anwendbarkeit auf das LPPL-System des Users (`lpplattr02_ode.py`)
10. Was der User MATLAB-seitig tun müsste — konkrete Checkliste

---

## 1. Was ist SSMtool und was ist eine Spectral Submanifold

### 1.1 Definition (Haller & Ponsioen 2016)

Eine **Spectral Submanifold (SSM)** über einem Spektral-Subraum $\mathcal{E}$ eines hyperbolischen Fixpunktes ist die **glatteste, eindeutige nichtlineare Fortsetzung** dieses Subraums. Sie ist invariant unter dem Fluss, tangential an $\mathcal{E}$ im Fixpunkt, und ihre Glattheit ist höher als jede andere invariante Mannigfaltigkeit, die $\mathcal{E}$ tangential berührt.

Formal: Sei $\dot{x} = f(x)$ ein autonomes glattes ODE-System mit asymptotisch stabilem Fixpunkt $x_0$, $A = Df(x_0)$ semisimpel. $\mathcal{E} \subset \mathbb{R}^n$ sei ein $A$-invarianter Spektral-Unterraum (aufgespannt von einer Teilmenge der Eigenvektoren). Eine SSM $\mathcal{W}(\mathcal{E})$ ist eine glatte invariante Mannigfaltigkeit, sodass:
- $T_{x_0}\mathcal{W}(\mathcal{E}) = \mathcal{E}$
- $\mathcal{W}(\mathcal{E})$ ist klassen-$C^r$ glatt mit $r$ größer als der Smoothness aller anderen $\mathcal{E}$-tangentialen invarianten Mannigfaltigkeiten (Eindeutigkeit modulo dieser Smoothness-Bedingung)
- Existenz und Eindeutigkeit folgen aus dem **Cabré-Fontich-de la Llave-Theorem** (CFdlL, 2003), das ein Spektralgap und Non-Resonanz fordert.

### 1.2 Was SSMtool konkret macht

SSMtool ist eine MATLAB-Implementierung der **Parametrization Method** zur Berechnung **zweidimensionaler** SSMs in autonomen, dissipativen mechanischen Systemen mit beliebig vielen Freiheitsgraden. Die SSM wird als formale Taylor-Reihe um den Fixpunkt konstruiert; die reduzierte Dynamik auf der SSM ergibt ein 2D-ODE in Polar-Koordinaten, aus dem **Backbone-Kurven** (Amplitude vs. instantane Frequenz) abgelesen werden.

Manual S. 4 (abridged, Auslassungen explizit markiert):
> "SSMtool is a Matlab-based computational tool for computing two-dimensional spectral submanifolds (SSMs) in nonlinear mechanical systems with arbitrary degrees of freedom. […Sätze über mathematischen Hintergrund ausgelassen…] The software achieves this without using any numerical integration or numerical continuation techniques, purely based on a reduction to SSMs."
> — `SSMtool_manual.pdf:4`

Manual S. 4:
> "Under appropriate non-resonance conditions, we can construct a two-dimensional autonomous SSM, $\mathcal{W}(\mathcal{E})$, over a chosen spectral subspace $\mathcal{E}$ as an embedding of a two-dimensional open set $\mathcal{U}$ into the full phase space of the system via a mapping $W(z)$. This mapping is approximated in a neighborhood of the origin using a Taylor expansion in the parametrization coordinates $z \in \mathbb{C}^2$."
> — `SSMtool_manual.pdf:4`

### 1.3 Unterschied SSMtool vs. SSMLearn

| Eigenschaft | SSMtool (Ponsioen & Haller 2017) | SSMLearn (Haller-Gruppe ab 2021) |
|---|---|---|
| Datenquelle | analytisches/symbolisches ODE-Modell, $M, C, K, f_{nl}$ | Trajektoriendaten (numerisch oder Experiment) |
| Methode | Parametrization Method, Polynomiale Reihen | Manifold Learning + Polynom-Regression |
| Modell-Voraussetzung | Mechanisches System $M\ddot{y} + C\dot{y} + Ky + g(y,\dot y) = 0$ explizit | Keine — nur Trajektorien |
| Output | Symbolische SSM-Polynome $W(z)$, $R(z)$, Backbone | Numerische SSM-Polynome, FRC, Backbone |

SSMtool **ist nicht data-driven**. Wer nur Trajektorien hat, muss SSMLearn benutzen.

Ergänzung 2026:
- Das lokal vorhandene Repo `/home/hz/Data/Attractor/globalized-SSM/` sitzt methodisch **hinter** beiden Ansätzen.
- Es benutzt vorhandene lokale SSM-Repräsentationen, um sie per **Padé-Approximation** oder **rationaler Regression** zu globalisieren.
- Für SSMtool bedeutet das konkret: die aus der Parametrization Method erhaltenen Taylor-Koeffizienten sind nicht zwingend Endprodukt, sondern können als Ausgangsbasis für eine globalere rationale Darstellung dienen.

### 1.4 SSMtool, SSMtool 2.0 und Addendum_Isolas

- **SSMtool V1.0 (2017)**: Nur autonome 2D-SSMs für mechanische Systeme. GUI-getrieben.
- **Addendum_Isolas (Dezember 2018)**: MATLAB-Skripte (KEIN GUI) zur Berechnung **zeit-periodisch geforcierter** 2D-SSMs (autonomer Anteil $W_0$ + linearer Anteil $W_1$ in der Forcierungs-Amplitude $\epsilon$), Forced Response Curves (FRC) und **Isolas** (isolierte Antwortkurven).
- **SSMtool 2.0**: im Manual als „kommendes Release 2019" angekündigt, im Repo nicht enthalten.

Manual S. 14:
> "We would like to note that the future release of SSMtool will be able to handle the time-dependent periodic forcing case as explained in [3]. This will allow you to extract forced response curves corresponding to vibration modes of interest for different forcing amplitudes, in an even more numerically efficient way. Additionally, and due to the use of SSM theory, it will be possible to detect isolated regions in the forced response curves."
> — `SSMtool_manual.pdf:14`

---

## 2. Mathematischer Rahmen — vollständig ausgeschrieben

### 2.1 Mechanisches System (Eingangsform)

SSMtool erwartet ein autonomes $n$-Freiheitsgrad-System der Form:

$$M\ddot{y} + C\dot{y} + K y + g(y,\dot{y}) = 0, \qquad g(y,\dot{y}) = \mathcal{O}(|y|^2,|y||\dot y|,|\dot y|^2)$$

mit:
- $y \in \mathbb{R}^n$ — generalisierter Lagevektor
- $M = M^T \in \mathbb{R}^{n\times n}$ — positiv definite Massenmatrix
- $C = C^T \in \mathbb{R}^{n\times n}$ — Dämpfungsmatrix
- $K = K^T \in \mathbb{R}^{n\times n}$ — Steifigkeitsmatrix
- $g$ — analytische Nichtlinearität, beginnt mit Ordnung 2.

(Quelle: Manual S. 4, Gleichung (1.1).)

> "where $y \in \mathbb{R}^n$ is the generalized position vector; $M = M^T \in \mathbb{R}^{n \times n}$ is the positive definite mass matrix; $C = C^T \in \mathbb{R}^{n \times n}$ is the damping matrix; $K = K^T \in \mathbb{R}^{n \times n}$ is the stiffness matrix and $g(y,\dot{y})$ denotes all the nonlinear terms in the system. These nonlinearities are assumed to be analytic for simplicity. Additionally, we require that the trivial fixed point of system (1.1) is asymptotically stable."
> — `SSMtool_manual.pdf:4`

### 2.2 First-order-Form (intern)

Mit $x = (y, \dot y)^T \in \mathbb{R}^{2n}$ wird das System auf erste Ordnung gebracht:

$$\dot{x} = A x + F_{nl}(x), \qquad A = \begin{pmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{pmatrix}, \qquad F_{nl}(x) = \begin{pmatrix} 0 \\ -M^{-1} g(y,\dot y) \end{pmatrix}.$$

Im Code (`compute_subspace.m:3`):
```matlab
A = [sym(zeros(size(M))),sym(eye(size(M)));-(M\K),-(M\C)];
```

In `SSM.m:409`:
```matlab
fnl = [sym(zeros(numel(q),1));-M\f];
```

### 2.3 Voraussetzungen für die Existenz der SSM (Cabré-Fontich-de la Llave, 2003)

Für einen autonomen, glatten Fluss $\dot x = f(x)$ mit $f(x_0)=0$ und $A = Df(x_0)$ existiert über jedem $A$-invarianten Spektral-Unterraum $\mathcal{E}$ eine eindeutige glatte invariante Mannigfaltigkeit, sofern die folgenden Bedingungen alle erfüllt sind:

#### (A) Hyperbolizität / asymptotische Stabilität
$\mathrm{Re}(\lambda_i) < 0$ für alle Eigenwerte $\lambda_i$ von $A$.

Im Code (`compute_subspace.m:40-44`):
```matlab
if ~conservative 
    if ~isempty(lambda(real(lambda)>=0))
        error('Real part for each eigenvalue of Spec(A) must be strictly negative')
    end
end
```

D. h. SSMtool **bricht ab**, wenn auch nur ein einziger Eigenwert nicht-negativen Realteil hat. Sattel, Zentren, Limit-Cycles, instabile Fixpunkte sind ausgeschlossen.

#### (B) Spektralgap zwischen Master und Rest
Sei $\mathcal{E}$ aufgespannt von zwei (komplex-konjugierten) Eigenwerten $\lambda_1, \lambda_2$. Definiere den **äußeren Spektral-Quotienten**

$$\sigma(\mathcal{E}) = \mathrm{Int}\!\left[\frac{\min_{k\notin E}\mathrm{Re}\,\lambda_k}{\max_{i\in E}\mathrm{Re}\,\lambda_i}\right] = \left\lfloor \frac{\mathrm{Re}\,\lambda_{\text{slow,outer}}}{\mathrm{Re}\,\lambda_{\text{fast}}}\right\rfloor.$$

Im Code (`check_res.m:14`):
```matlab
sigma = double(floor(real(lambda_remain(end))/real(lambda_select(1))));
```
$\sigma$ gibt die Mindestordnung der Taylor-Entwicklung an, die nötig ist, damit die SSM eindeutig wird (alle Polynom-Beiträge unter $\sigma$ sind durch andere $\mathcal{E}$-tangentiale Mannigfaltigkeiten nicht festgelegt).

Sicherheits-Cap (`check_res.m:15-18`):
```matlab
if sigma > 50
    sigma = 5; 
    uiwait(msgbox('Large outer spectral quotient detected, sigma is set to a default value of 5.','Setting Spectral Quotient','warn'));
end
```

#### (C) Äußere Non-Resonanz (External Non-Resonance Condition)

Für alle $k \notin E$ und alle Multi-Indizes $(m_1, m_2) \in \mathbb{N}_0^2$ mit $2 \le m_1+m_2 \le \sigma(\mathcal{E})$:

$$\lambda_k \neq m_1 \lambda_1 + m_2 \lambda_2.$$

Im Code wird die schwächere, numerisch robustere Bedingung
$$\frac{|m_1\lambda_1 + m_2\lambda_2 - \lambda_k|}{\|(m_1,m_2,-1)\|\,\|(\lambda_1,\lambda_2,\lambda_k)\|} \ge 10^{-4}$$
geprüft. Die Cosinus-ähnliche Resonanzfunktion ist in `check_res.m:9-11`:
```matlab
syms a b lambda_1 lambda_2 lambda_k z1 z2
res = abs(([a;b;-1].'*[lambda_1;lambda_2;lambda_k])/(norm([a;b;-1])*norm([lambda_1;lambda_2;lambda_k])));
matlabFunction(res,'file','res_function','Vars',[a;b;lambda_1;lambda_2;lambda_k]);
```
Die Toleranz für externe Resonanz ist $10^{-4}$ (`check_res.m:24`):
```matlab
if double(res_function(ord(m,1),ord(m,2),lambda(getValues(1)),lambda(getValues(2)),lambda(k))) < 1e-4  
```
Bei Verletzung wird abgebrochen:
```matlab
txt = 'The external nonresonance conditions are violated.';
h = errordlg(txt,'External resonance detected',mode);
```

#### (D) Innere Resonanzen (Internal Resonance / Near-Inner-Resonance)
Falls $\lambda_1, \lambda_2 \in E$ und es Multi-Indizes $(m_1,m_2)$ gibt mit
$$\lambda_l \approx m_1 \lambda_1 + m_2 \lambda_2 \quad \text{für } l \in \{1,2\},$$
dann ist die Normalform NICHT die triviale lineare Form, sondern enthält **resonante Terme**, die in $R(z)$ verbleiben statt in $W(z)$ wegtransformiert zu werden. SSMtool prüft das mit Toleranz $5 \cdot 10^{-2}$ (`check_res.m:45`, `check_higher_res.m:19`):
```matlab
if double(res_function(ord(m,1),ord(m,2),lambda(getValues(1)),lambda(getValues(2)),lambda(k))) < 5e-2  
```
Wenn eine innere Resonanz erkannt wird, wird das Flag `int_res = 1` gesetzt und die resonanten Polynom-Koeffizienten werden in `loc_R` gespeichert. Diese werden in der Cohomologischen Gleichung anders behandelt.

#### (E) Glattheit
$f$ analytisch (oder mindestens $C^r$ für genügend großes $r$). Manual fordert explizit Analytizität (S. 4: „These nonlinearities are assumed to be analytic for simplicity.").

### 2.4 Parametrization Method — Invariance Equation

Wir suchen eine Einbettung $W: \mathbb{C}^2 \supset \mathcal{U} \to \mathbb{R}^{2n}$ und eine reduzierte Dynamik $R: \mathbb{C}^2 \to \mathbb{C}^2$, sodass die **Invarianzgleichung**

$$\boxed{\;DW(z)\,R(z) \;=\; F(W(z))\;}$$

für alle $z \in \mathcal{U}$ gilt. Hier ist $F$ das vollständige rechte-Seite-Vektorfeld in First-order-Form:
$$F(x) = Ax + F_{nl}(x).$$

Geometrisch heißt das: Der Vektor $F$ am Punkt $W(z)$ liegt im Tangentialraum an $\mathcal{W}(\mathcal{E})$, dargestellt durch $DW(z)\cdot R(z)$. Die SSM ist also invariant.

Die reduzierte Dynamik auf der SSM ist
$$\dot z = R(z), \qquad z \in \mathbb{C}^2.$$

Mit $z = \rho e^{i\theta}, \bar z = \rho e^{-i\theta}$ wird das Polar-System
$$\dot \rho = a(\rho), \qquad \dot \theta = b(\rho).$$
(Manual S. 4, Gleichung (1.3); siehe auch `SSM_exp.m:140-142`.)

> "By introducing a change to polar coordinates, eq. (1.2) can be rewritten as $\dot{\rho} = a(\rho), \quad \Omega = \dot{\theta} = b(\rho),$ where we refer to section 5.2 of Ponsioen et al. [2] for a detailed explanation of this derivation."
> — `SSMtool_manual.pdf:4`

Die **Backbone-Kurve** ist die Beziehung
$$\Omega(\rho_0) = b(\rho_0)$$
ausgewertet an verschiedenen Amplituden $\rho_0$. Die instantane Frequenz hängt von der Amplitude ab — das ist die Definition der nichtlinearen Resonanzkurve eines dissipativen Systems.

### 2.5 Polynomieller Ansatz — formale Reihen

Schreibe $W$ und $R$ als Taylor-Reihen in $z = (z_1, z_2)$:
$$W(z) = \sum_{|m|\ge 1} W_m\, z_1^{m_1} z_2^{m_2}, \qquad R(z) = \sum_{|m|\ge 1} R_m\, z_1^{m_1} z_2^{m_2}.$$

Die linearen Anteile sind festgelegt:
- $W_1 = (T_E\;|\;0)$, wo $T_E$ aus den beiden Eigenvektoren zu $\lambda_1, \lambda_2$ besteht. Im Code (`compute_SSM.m:47`): `K1 = [eye(2);zeros(sys_dim-2,2)];` (im modal-projizierten Raum).
- $R_1 = \mathrm{diag}(\lambda_1, \lambda_2)$. Im Code (`compute_SSM.m:46`): `R1 = diag(sys.lambda.num(sys.lambda_select));`

Die höheren Koeffizienten $W_m, R_m$ ($|m| \ge 2$) werden ordnungsweise aus den **cohomologischen Gleichungen** bestimmt.

### 2.6 Cohomologische Gleichungen (ordnungsweise)

Setze die Reihen in $DW \cdot R = F \circ W$ ein und sortiere nach Polynomgrad. Bei Grad $k$ ergibt sich:

$$\underbrace{A\, W_k - W_k\, \mathcal{R}_k}_{\text{linearer Operator } \mathcal{L}_k} \;=\; \mathrm{RHS}_k(W_1,\ldots,W_{k-1}, R_1,\ldots,R_{k-1})$$

Die rechte Seite $\mathrm{RHS}_k$ enthält ausschließlich niedrigerordige Größen (deshalb "ordnungsweise lösbar"). Sie besteht aus zwei Beiträgen:
- $K_R = \sum_{m=2}^{k-1} (\partial_z W_m) R_{k+1-m}$ — Kontribution aus dem Differential von $W$ (`kronKR.m`)
- $G_K = \sum_{m=2}^{k-1} G_m \cdot \mathrm{Sym}(W_{?})$ — Kontribution aus der Komposition der Nichtlinearität mit $W$ (`kronGK.m`, `kronGK1n.m`)

Dabei ist $G_m \in \mathbb{R}^{2n \times (2n)^m}$ die ausgepackte $m$-te Taylor-Stufe von $F_{nl}$ in modaler Basis (gebaut von `matGV2.m`).

Im Code (`compute_SSM.m:104-150`):
```matlab
free_dofs = 1:sys_dim*2^n;
c = combinator(2,n,'p','r');
RnI = sum(lambda_order(c),2);
At_vec = repmat(diag(At),[1,2^n]);
Ae_vec = repmat(RnI.',[sys_dim,1]);
LHS_n = reshape(At_vec-Ae_vec,[],1);  
KR = zeros(sys_dim,2^n);
G_check = G{n};
G_check_bool = numel(G_check(:))>1;
if  G_check_bool
    GK1 =  kronGK1n(n,K,G);
end
GK = zeros(sys_dim,2^n);
for m = 2:n-1
    KR = KR + kronKR(n,m,K,R);
    GK = GK + kronGK(n,m,K,G);  
end
...
if G_check_bool
    RHS_n = K1*R{n} + KR - GK1 - GK;  
else
    RHS_n = K1*R{n} + KR - GK;   
end
```

`LHS_n` ist der **Diagonalanteil** des cohomologischen Operators $\mathcal{L}_k$. Konkret ist das der Vektor mit Einträgen
$$\mathcal{L}_{k,j,(m_1,m_2)} = \mathrm{diag}(A)_j - (m_1\lambda_1 + m_2\lambda_2),$$
wo $j$ der modale Index $1\ldots 2n$ und $(m_1,m_2)$ der Polynom-Index sind. Diese Diagonalstruktur entsteht, weil SSMtool das System zuerst in die modale Basis transformiert (`compute_SSM.m:14`: `At = sys.At;` ist die Diagonalisierung von $A$).

Die Lösung pro freiem DOF ist also die elementweise Division (`compute_SSM.m:160-164`):
```matlab
spmd 
    range = id(labindex)+ 1:id(labindex+1);  
    sol_con = P_vec(range)./K_diag(range);   
end
K_n_vec_full(free_dofs) = cat(1,sol_con{:});
K_n = reshape(K_n_vec_full,[sys_dim,2^n]);
K{n} = K_n;
```

`spmd` ist MATLABs Single-Program-Multiple-Data-Block für Parallel-Toolbox: Die Polynomkoeffizienten werden auf die verfügbaren CPU-Kerne aufgeteilt.

### 2.7 Behandlung resonanter Terme (Normalform-Style)

Für jeden Polynomindex $(m_1, m_2)$ und jede Master-Mode $j \in E$ gilt: Wenn der Nenner
$$\mathcal{L}_{k,j,(m_1,m_2)} = \lambda_j - m_1\lambda_1 - m_2\lambda_2$$
nahe null ist, ist die cohomologische Gleichung degeneriert. SSMtool verschiebt diesen Beitrag dann **vom $W$-Polynom in das $R$-Polynom**: $W_{k,j,(m_1,m_2)} := 0$ und stattdessen wird $R_{k,j,(m_1,m_2)} = -K_R + G_K - G_{K_1}$ gesetzt (`compute_SSM.m:122-144`):
```matlab
if int_res
    Rn_dummy = R{n};
    KR_dummy = KR(1:2,:);
    if  G_check_bool
        GK1_dummy = GK1(1:2,:);      
    end
    GK_dummy = GK(1:2,:);
    for q = 1:2
        if numel(loc_R{q,n}) > 0                  
            if G_check_bool 
                Rn_dummy(q,loc_R{q,n}) = -KR_dummy(q,loc_R{q,n}) + GK1_dummy(q,loc_R{q,n}) + GK_dummy(q,loc_R{q,n});
            else 
                Rn_dummy(q,loc_R{q,n}) = -KR_dummy(q,loc_R{q,n}) + GK_dummy(q,loc_R{q,n});
            end     
        end
    end
    R{n} = Rn_dummy;
    dof = 1:sys_dim*2^n;
    constraints_u =  (loc_R{1,n}-1)*sys_dim+1;
    constraints_l =  (loc_R{2,n}-1)*sys_dim+2;
    constraints = union(constraints_u, constraints_l);
    free_dofs = setdiff(dof,constraints);
end
```
Damit wird die reduzierte Dynamik **automatisch in Normalform** geschrieben: $R(z)$ enthält alle resonanten Terme (Backbone, nichtlineare Frequenzkorrekturen), $W(z)$ enthält den Rest (Geometrie der SSM).

**Wichtig**: SSMtool benutzt also einen **gemischten Normal-Form-Style**: Wenn keine Resonanz, dann reine Graph-Form ($R$ bleibt linear); wenn Resonanz, dann werden die resonanten Terme in $R$ verschoben. Bei lightly-damped Systemen (typisch für Mechanik) gibt es fast immer near-inner-Resonanzen, deshalb ist der Defaultpfad fast immer der Normal-Form-Style.

### 2.8 Backbone-Kurven aus der Polar-Form

Mit $z_1 = \rho \cos\theta + i \rho \sin\theta = \rho e^{i\theta}$ (oder als Realwert-Paar) folgt aus $\dot z = R(z)$:
$$\dot\rho = \alpha(\rho)\cos\theta + \beta(\rho)\sin\theta, \qquad \dot\theta = \frac{1}{\rho}(\beta(\rho)\cos\theta - \alpha(\rho)\sin\theta)$$
(`SSM_exp.m:141-142`):
```matlab
R{1} = expand(simplify(alpha*cos(theta) + beta*sin(theta)));
R{2} = expand(simplify((1/r)*(beta*cos(theta)-alpha*sin(theta))));   
```
Wenn die instantane Frequenz $\dot\theta$ unabhängig von $\theta$ ist (d. h. nur von $\rho$ abhängt), wird sie als Funktion $\Omega(\rho_0)$ ausgewertet — das ist die **Backbone-Kurve**.

`SSM_exp.m:144-153`:
```matlab
if numel(symvar(jacobian(R{2},theta)))>1
    handles.back_avg = 0;
    handles.mono_var_freq  = 0;
    set(handles.checkbox_avg,'Value',0);
    set(handles.checkbox_avg,'Enable','Off');
    set(handles.popup_cor_avg,'Enable','On');
else   
    handles.mono_var_freq  = 1;
end  
```

### 2.9 Forced Response und Isolas (Addendum_Isolas)

Für ein Forcing $\epsilon F_\phi(\Omega t)$ wird das System
$$\dot x = Ax + F_{nl}(x) + \epsilon F_\phi(\Omega t)$$
auf eine **zeit-periodische SSM** reduziert. Der Ansatz ist
$$W(z, \phi) = W_0(z) + \epsilon W_1(z, \phi) + \mathcal{O}(\epsilon^2), \qquad R(z, \phi) = R_0(z) + \epsilon R_1(z, \phi) + \mathcal{O}(\epsilon^2),$$
mit $\phi = \Omega t \in S^1$. Die Invarianzgleichung in erster Ordnung in $\epsilon$ wird:

$$DW_0(z) R_1(z,\phi) + DW_1(z,\phi) R_0(z) + \partial_\phi W_1(z,\phi)\,\Omega \;=\; D F_{nl}(W_0(z))\,W_1(z,\phi) \,+\, F_\phi(\phi) \,+\, A\,W_1(z,\phi).$$

Im Code des Beispiels `ex_SP_W0_3_W1_0.m:407-410`:
```matlab
% Invariance equation: A*W1 + \partial_x{G(W0)}*W1 + F(phi) =
% dW0ds*R1 + dW1ds*R0 + dW1dphi*\Omega
```

Mit Fourier-Ansatz $W_1(z,\phi) = W_{1,a}(z)e^{i\phi} + W_{1,b}(z)e^{-i\phi}$ ergibt sich pro Mode $j$ und Polynomindex $(m_1,m_2)$ der Nenner
$$\lambda_j - (m_1\lambda_1 + m_2\lambda_2) - i\Omega \quad \text{(für } e^{i\phi}\text{-Anteil)},$$
$$\lambda_j - (m_1\lambda_1 + m_2\lambda_2) + i\Omega \quad \text{(für } e^{-i\phi}\text{-Anteil)}.$$
(`ex_SP_W0_3_W1_0.m:431-432`):
```matlab
den_a = lambda(l) - index_corder(j,:)*lambda_E-1j*omega;    
den_b = lambda(l) - index_corder(j,:)*lambda_E+1j*omega;  
```

Die FRC entsteht als **implizite** Beziehung $G(\rho, \Omega) = 0$ aus dem Verschwinden der gemittelten reduzierten Dynamik:
$$\rho\,(b(\rho)-\Omega) + \epsilon(g_1(\rho)\cos\psi - g_2(\rho)\sin\psi) = 0,$$
$$a(\rho) + \epsilon(f_1(\rho)\cos\psi + f_2(\rho)\sin\psi) = 0.$$
(`ex_SP_W0_5_W1_5.m:666-674`):
```matlab
rhod =  a + epsilon*(f1*cos(psi) + f2*sin(psi));
psid =  (b-omega) + (epsilon/rho)*(g1*cos(psi) - g2*sin(psi));

disc = epsilon^2*(f1^2+f2^2)-a^2;
Kp = (-epsilon*f2 + sqrt(disc))/(a-epsilon*f1);
Km = (-epsilon*f2 - sqrt(disc))/(a-epsilon*f1);

Fimp_p = (b-omega).*rho + epsilon.*(g1.*(1-Kp.^2)./(1+Kp.^2)-g2.*(2.*Kp)./(1+Kp.^2));
Fimp_m = (b-omega).*rho + epsilon.*(g1.*(1-Km.^2)./(1+Km.^2)-g2.*(2.*Km)./(1+Km.^2));
```

**Isolas** (isolierte Antwortkurven, die nicht mit dem Hauptzweig zusammenhängen) werden durch Konturplot von $G(\rho, \Omega) = 0$ in der $(\Omega, \rho)$-Ebene detektiert (`ex_SP_W0_5_W1_5.m:764-766`):
```matlab
[C1,h1] = contour(Yomega,Xrho,Fimpeval_m,[0,0],'-','LineWidth',1.5,'Color',[255/255 0/255 51/255]);
[C2,h2] = contour(Yomega,Xrho,Fimpeval_p,[0,0],'-','LineWidth',1.5,'Color',[255/255 0/255 51/255]);
```

Stabilität wird per Jacobi der gemittelten Dynamik berechnet (Code-Abschnitte ab `ex_SP_W0_5_W1_5.m:687`).

---

## 3. Inputs, die SSMtool zwingend erwartet

### 3.1 Form: mechanisches System mit $M, C, K, f$

SSMtool akzeptiert NUR mechanische Systeme zweiter Ordnung. Es erwartet (Manual S. 8, Tabelle 2.1):

| Variable | Name | Klasse | Bedeutung |
|---|---|---|---|
| Massenmatrix | `M` | `n × n double` | $M$ |
| Dämpfungsmatrix | `C` | `n × n double` | $C$ |
| Steifigkeitsmatrix | `K` | `n × n double` | $K$ |
| Nichtlinearer Kraftvektor | `f` | `n × 1 sym` | $g(y,\dot y)$ — symbolisch in $x_1,\ldots,x_n,\dot x_1,\ldots,\dot x_n$ |
| Lagevektor | `x` | `n × 1 sym` | Symbol-Liste $\{x_1,\ldots,x_n\}$ |
| Geschwindigkeitsvektor | `xd` | `n × 1 sym` | Symbol-Liste $\{\dot x_1,\ldots,\dot x_n\}$ |
| Conservative-Flag | `conservative` | `int oder double` | $0$ = nein (mit Dämpfung), $1$ = ja (rein konservativ) |
| Eigenvektor-Skalierung | `scaling` | `double` | Skalierungs-Faktor für die Eigenvektor-Spalten von $T$ |

Diese Felder werden via `.mat`-Datei eingeladen (`SSM.m:1160`):
```matlab
if isfield(struct,'M') && isfield(struct,'C')  && isfield(struct,'K') && isfield(struct,'f')  && isfield(struct,'x') && isfield(struct,'xd') && isfield(struct,'scaling') && isfield(struct,'conservative')
```

**Alternativ** kann der User die Matrizen direkt im GUI eintippen (`SSM.m:380-407`).

### 3.2 Was SSMtool NICHT akzeptiert
- Generelle First-Order-ODEs $\dot x = f(x)$ ohne mechanische Struktur — müssen erst in $M\ddot y + \ldots$ gebracht werden.
- Numerische $f$-Funktionshandle — die Nichtlinearität muss **symbolisch** angegeben werden (für die Taylor-Entwicklung via `taylor()` in `compute_SSM.m:25,34`).
- Zeitabhängige $f(x,t)$ — V1 ist autonom-only.
- Nicht-triviale Fixpunkte. SSMtool prüft am Anfang von `compute_SSM.m:16-22`:
```matlab
check_eq = abs(sum(subs(sys.f,sys.spv,zeros(numel(sys.spv),1))));
...
if check_eq > 1e-10
    errordlg('The equilibrium point of the system is non-trivial','Non-trivial equilibrium point', mode); 
    return;
end
```
Der Fixpunkt MUSS am Ursprung liegen. Wenn nicht, muss der User VOR Übergabe an SSMtool $x \to x - x_0$ verschieben.

### 3.3 Auswahl-Inputs aus dem GUI
- **Master-Spektral-Subraum**: zwei Eigenwerte aus der Liste der berechneten $\lambda$, manuell vom User markiert (`SSM.m:482`, `push_select_lambda_Callback`). Mehr als 2 oder weniger als 2 sind nicht erlaubt:
```matlab
if numel(getValues) < 2    
    h = errordlg('Please select two eigenvalues.','Selection Error', mode);
```
Die zwei gewählten Eigenwerte müssen entweder beide reell oder ein komplex-konjugiertes Paar bilden (`orderT.m:81-87`):
```matlab
else
    txt =  'Cannot form a 2D spectral subspace for the chosen set of eigenvalues.';
    h = errordlg(txt,'Selection Error', mode);
```
- **Order of SSM expansion**: Integer von 2 bis maximal 50 (`SSM.m:575-579`):
```matlab
elseif handles.sys.n>50
    txt = 'The maximum order of SSM expansion is currently set to 50';
```
Wenn der Spektral-Quotient $\sigma \ge n$ ist, wird abgebrochen mit:
```matlab
txt = 'Chosen order of SSM expansion is lower than the spectral quotient plus one.';
```
- **Output-Koordinaten**: `Modal` / `Physical` / `Complex` — Radio-Buttons. Bestimmt, in welcher Basis die Output-Polynome ausgegeben werden.
- **Check for higher-order internal resonances**: Checkbox; wenn gesetzt, werden Resonanz-Tests bis zur SSM-Ordnung statt nur bis $\sigma$ gemacht.

---

## 4. Pipeline Schritt für Schritt

Diese Sektion ist die **Kernreferenz**. Jeder Schritt referenziert die ausführende Funktion und die wichtigsten Code-Zeilen.

### Schritt 0: GUI-Initialisierung
- Datei: `SSM.m`, Funktion: `SSM_OpeningFcn` (`SSM.m:48-117`)
- Aktion: GUI-Layout, Pfad zu `Data/`-Subfolder.
- Befehl zum Start: `SSM` in der MATLAB-Konsole (Manual S. 6).

### Schritt 1: System einladen
- Drei Wege:
  1. Predefined example via Dropdown (`popup_load_example_Callback`, `SSM.m:1590`) — lädt `Examples/2DOF_inner_res.mat`, `Examples/2DOF_outer_res.mat` oder `Examples/beam.mat`.
  2. Custom `.mat`-File via Browse-Button (`push_loadmech_Callback`, `SSM.m:1128`).
  3. Manuelle GUI-Eingabe der Matrizen (Felder `input_mass`, `input_damping`, `input_stiffness`, `input_nonlinear`, `input_pos`, `input_vel`).
- Manual S. 6:
(abridged, Auslassungen explizit markiert)
> "To load an autonomous mechanical system into SSMtool, there is a total of three options. The first option is to load a predefined mechanical system from the drop-down list in the top-left corner of the GUI […Detailtext zu Option 1 ausgelassen…] The second option is to define your mechanical system in the predefined input windows […Detailtext zu Option 2 ausgelassen…] The third and last option available is to load a custom mechanical system from a .mat file containing the variables listed in Table 2.1"
> — `SSMtool_manual.pdf:6-7`

### Schritt 2: Linearisierung und Eigendecomposition (`Analyze`-Button)
- Datei: `compute_subspace.m`
- Bildet First-order-Matrix $A$:
```matlab
A = [sym(zeros(size(M))),sym(eye(size(M)));-(M\K),-(M\C)];
```
- `[Xmode,lambda] = eig(A)` — symbolische Eigendecomposition. Wenn $A$ nicht semisimpel (nicht diagonalisierbar) ist, Fehler:
```matlab
if rank(Xmode) < size(A,1)
   error('Linear matrix A is not semisimple')
end
```
- Sortierung:
  - Konservativ: aufsteigend nach Imaginärteil (`compute_subspace.m:12-22`).
  - Nicht-konservativ: nach $-\mathrm{Re}\lambda$, dann $-\mathrm{Im}\lambda$ (`compute_subspace.m:24-26`):
```matlab
col   = [real(lambda),imag(lambda)];
[~,k] = sortrows(double(col),[-1,-2]);
```
- Eigenvektoren werden auf Einheits-Norm normiert und mit `scaling` skaliert.
- **Hard Check**: Alle Eigenwerte müssen $\mathrm{Re}\lambda < 0$ haben — sonst Abbruch.
- Output: $\lambda$, $T$ (= Matrix der Eigenvektoren), $A$ (numerisch).

### Schritt 3: Master-Subraum-Selektion
- Funktion: `push_select_lambda_Callback` (`SSM.m:482`)
- User wählt 2 Eigenwerte aus der Liste. Die übrigen $2n-2$ Eigenwerte werden in `lambda_remain` gespeichert.
- Funktion: `orderT.m` ordnet die Spalten von $T$ so um, dass die ersten 2 Spalten den Master-Subraum aufspannen. Die übrigen folgen.
- Bei komplex-konjugiertem Master-Paar wird `complex_cor = 1` gesetzt; bei reellem Paar `complex_cor = 0`.
- Bilde $\tilde A = T^{-1} A T$ — modale Matrix, idealerweise diagonal:
```matlab
At = T\A*T;   
handles.sys.At = At; 
```

### Schritt 4: Nicht-Resonanz-Check
- Funktion: `check_res.m`
- (a) Berechne $\sigma(\mathcal{E})$ aus dem Verhältnis von größtem zu kleinstem Realteil.
- (b) Iteriere über alle Polynom-Ordnungen $l = 2,\ldots,\sigma$ und alle $(m_1,m_2)$ mit $m_1+m_2=l$, alle $\lambda_k \in$ remain. Wenn die Resonanz-Funktion $< 10^{-4}$, breche ab (externe Resonanz).
- (c) Iteriere für innere Resonanzen mit Toleranz $5\cdot 10^{-2}$ und speichere alle resonanten Multi-Indizes in `nonlin_vec` (`int_res_vec`).
- Funktion `check_higher_res.m` (optional, wenn Checkbox gesetzt): wiederhole für $l = \sigma+1,\ldots,n$ (Polynomordnung der SSM-Entwicklung).

### Schritt 5: SSM berechnen (`Compute SSM`-Button)
- Funktion: `push_compute_Callback` (`SSM.m:560`) bereitet die `sys`-Struktur vor und ruft `compute_SSM(sys, n)`.
- Hauptfunktion: `compute_SSM.m`
- Schritte innerhalb von `compute_SSM`:
  1. **Trivialität des Fixpunktes prüfen** (`compute_SSM.m:16`): siehe oben.
  2. **Nichtlinearität bei Ordnung 2 starten lassen**: alle linearen Anteile aus `f` entfernen (`compute_SSM.m:24-31`):
```matlab
for i = 1:numel(sys.spv)
   f(i) = simplify(taylor(sys.f(i),sys.spv.','ExpansionPoint',0,'Order',2)-taylor(sys.f(i),sys.spv.','ExpansionPoint',0,'Order',1));
end
if numel(find(f==0))<numel(sys.spv)
    errordlg('The Taylor expansion of the nonlinear force vector contains linear terms.','Order Error', mode); 
```
  3. **Taylor-Reihe von $f$ bis Ordnung $n+1$** (`compute_SSM.m:33-37`):
```matlab
for i = 1:numel(sys.spv)
    f(i) = simplify(taylor(sys.f(i),sys.spv.','ExpansionPoint',0,'Order',order+1));
end
f = expand(f);
```
  4. **Initialisiere $W_1$ und $R_1$** (`compute_SSM.m:46-49`):
```matlab
R1 = diag(sys.lambda.num(sys.lambda_select));
K1 = [eye(2);zeros(sys_dim-2,2)];
K = cell(1,order); R = cell(1,order); X = cell(1,order);
K{1} = K1; R{1} = R1; X{1} = X1;
```
  5. **Baue Polynom-Monom-Listen $X_n = \{z_1^{b_1}z_2^{b_2}\}$** für jede Ordnung (`compute_SSM.m:51-58`):
```matlab
for i = 2:order
    K{i} = zeros(sys_dim,2^i);
    R{i} = zeros(2,2^i);
    c = combinator(2,i,'p','r');
    b1 = sum(c==1,2);
    b2 = sum(c==2,2);
    X{i} = z1.^b1.*z2.^b2;
end
```
Hier wird `combinator(2,i,'p','r')` benutzt — Permutationen mit Repetition; das gibt $2^i$ Indextupel. Das ist eine **redundante** Darstellung (nicht $\binom{i+1}{1}$ wie nsumk), die aber mit den Kronecker-Produkt-Strukturen in `kronGK`, `kronKR`, `nkron` kompatibel ist.
  6. **Baue Nichtlinearitäts-Matrizen $G_m$** mit `matGV2(f, spv, T, order)` (`compute_SSM.m:61`).
     - `matGV2.m` liest aus dem symbolischen $f$ alle Monomial-Beiträge, identifiziert deren Ordnung, baut sparse Matrizen $G_m$ und transformiert sie in die modale Basis: `G{i} = T\Gcoef*nkron(i,T);` (`matGV2.m:76`).
     - Kommentar im Code: `% Building up matrices` (Statusmessage `matGV2.m:6`).
  7. **Lokalisiere innere Resonanzen** in `loc_R` (`compute_SSM.m:72-95`).
  8. **Schleife über Polynom-Ordnung $n=2,\ldots,\text{order}$** (`compute_SSM.m:100-169`):
     - Berechne den diagonalen Anteil $\mathcal{L}_n$.
     - Berechne $K_R = \sum_{m=2}^{n-1} \partial W_m \cdot R_{n+1-m}$ via `kronKR(n,m,K,R)`.
     - Berechne $G_K = \sum_{m=2}^{n-1} G_m \cdot \mathrm{Sym}(K_{?})$ via `kronGK(n,m,K,G)`.
     - Falls `G{n}` nicht null ist: zusätzlich $G_{K_1} = G_n \cdot K_1^{\otimes n}$ via `kronGK1n(n,K,G)`.
     - Behandle resonante Indizes: setze die entsprechenden $R_n$-Einträge gemäß Normalform-Style.
     - Lös den Rest der cohomologischen Gleichung elementweise: `sol_con = P_vec(range)./K_diag(range);` (parallel via `spmd`).
  9. **Aufsummieren** zur formalen Reihe (`compute_SSM.m:173-179`):
```matlab
for i = 1:order
    SSM_FPS = SSM_FPS + round(K{i},17)*X{i};
    R_FPS = R_FPS + round(R{i},17)*X{i};
end
```
Das `round(...,17)` ist ein numerischer Sweep, um numerischen Müll unter $10^{-17}$ wegzuwerfen.
  10. **Output-Projektion** in modal/physical/complex Koordinaten (`compute_SSM.m:181-187`):
```matlab
if modal
    SSM_proj = Tmodal\T*SSM_FPS;
elseif complex
    SSM_proj = SSM_FPS;
else
    SSM_proj = T*SSM_FPS;
end
```
  11. **Schreiben in MATLAB-Funktionsdateien** (`compute_SSM.m:208-218`):
```matlab
matlabFunction(input_vars_SSM{:},'file',strcat('SSM_function_',nm.folder_id),'Vars',X1,'Outputs',output_vars_SSM);
matlabFunction(input_vars_R{:},'file',strcat('R_function_',nm.folder_id),'Vars',X1,'Outputs',output_vars_R);
matlabFunction(input_vars_R{:},'file','R_sub_function','Vars',X1,'Outputs',output_vars_R);
```
Die Output-Funktion `SSM_function_<timestamp>.m` ist die berechnete $W(z_1,z_2)$, `R_function_<timestamp>.m` ist die berechnete $R(z_1,z_2)$.

### Schritt 6: Backbone-Curve-Extraktion
- Funktion: `SSM_exp_OpeningFcn` (`SSM_exp.m:48`) und `push_backbone_Callback` (`SSM_exp.m:207`).
- Konvertiere $R(z_1,z_2)$ in Polar-Form $\dot\rho = a(\rho), \dot\theta = b(\rho)$.
- Plot $\Omega(\rho_0) = b(\rho_0)$ vs. Amplitude $|x_i|_{\max}$ oder $\langle |x| \rangle$.
- Manual S. 11:
> "As explained in the introduction of this guide and in section 5.2 of Ponsioen et al. [2], the backbone curve can be extracted from the $\Omega(\rho) = \dot{\theta}(\rho)$ equation of the reduced system. By pressing the `Plot` button, SSMtool will compute the backbone curve in the physical coordinate system."
> — `SSMtool_manual.pdf:11`

### Schritt 7: Trajektorien-Integration und Vergleich
- Funktion: `SSM_int.m` (GUI), `int_red_dyn.m` (reduzierte Dynamik), `int_dyn.m` (vollständige Dynamik).
- Reduzierte Dynamik wird via `ode45` integriert (`int_red_dyn.m:14`):
```matlab
[tInter,ystateInter] = ode45(@(t,y) odefun(t,y),[0,t_end],y0(i,:),options);
```
mit RelTol $10^{-11}$, AbsTol $10^{-15}$. Ergebnis wird via $W(z)$ in den vollen Phasenraum zurückgemappt.
- Volle Dynamik wird via `ode15s` integriert (`int_dyn.m:17`).

### Schritt 8: Invarianz-Fehlermessung
- Funktion: `measure_inv_autonomous.m`
- Manual S. 14, Gleichung (2.1):
$$\delta_{\text{inv}} = \frac{1}{N}\sum_{i=1}^N \frac{\mathrm{dist}(i)}{\max_{\theta\in S^1}\|\tilde x(\rho_0,\theta)\|_2}, \qquad \mathrm{dist}(i) = \max\left\|x_i\big|^{\rho_\epsilon}_{\rho_0} - \tilde x_i\big|^{\rho_\epsilon}_{\rho_0}\right\|_2$$
- Es werden $N$ Trajektorien vom Kreis $\rho = \rho_0$ gestartet und integriert, bis sie den inneren Kreis $\rho_\epsilon < \rho_0$ erreichen. Die Distanz zwischen voller und reduzierter Trajektorie wird zum Fehler aggregiert.
- Code (`measure_inv_autonomous.m:67-78`):
```matlab
spmd 
range = id(labindex)+ 1:id(labindex+1);
[t_red,ystate_red,Rm,err,tevent] = int_red_dyn_em(tend,eps_init,y0(range,:),options,nm,s.sys);
end
```

---

## 5. Datei-für-Datei-Inventar

### 5.1 `SSMtool/SSMtool/` — Hauptverzeichnis (V1.0, GUI-getrieben)

#### GUI-Files
- **`SSM.m`** (1784 Zeilen) — Haupt-GUI. Enthält alle Callback-Funktionen für Buttons, Inputs, Dropdowns. Funktion `SSM` öffnet das GUI; `push_compute_Callback` ruft `compute_SSM` auf.
- **`SSM.fig`** — GUIDE-Layout-File für das Haupt-GUI.
- **`SSM_exp.m`** (419 Zeilen) — GUI für SSM/R-Expressions und Backbone-Plot.
- **`SSM_exp.fig`** — Layout.
- **`SSM_int.m`** (519 Zeilen) — GUI für Trajektorien-Integration.
- **`SSM_int.fig`** — Layout.
- **`SSM_invar.m`** (309 Zeilen) — GUI für Invarianz-Fehlermessung.
- **`SSM_invar.fig`** — Layout.

#### Kern-Computation
- **`compute_subspace.m`** (49 Zeilen) — Bildet $A$ in First-order-Form, eigendekomposition, Sortierung, Stabilitäts-Check.
  - Signatur: `function [lambda_out,T,A] = compute_subspace(M,C,K,scaling,conservative)`
- **`orderT.m`** (90 Zeilen) — Ordnet Eigenvektor-Spalten so, dass der Master-Subraum die ersten 2 Spalten besetzt; baut auch die modale Basis-Matrix `Tmodal` mit reellen Spalten für komplex-konjugierte Paare.
  - Signatur: `function [handles_out] = orderT(A,handles)`
- **`check_res.m`** (75 Zeilen) — Berechnet $\sigma(\mathcal{E})$, prüft externe und interne Resonanz bis Ordnung $\sigma$.
  - Signatur: `function [handles_out] = check_res(handles)`
- **`check_higher_res.m`** (90 Zeilen) — Prüft Resonanzen bis zur Polynom-Ordnung der SSM-Entwicklung (höher als $\sigma$).
  - Signatur: `function [handles_out,sys_out] = check_higher_res(handles,sys)`
- **`compute_SSM.m`** (224 Zeilen) — **Hauptfunktion**, löst die cohomologischen Gleichungen ordnungsweise. Schreibt SSM- und R-Funktionen als `.m`-Files in `Data/run_<timestamp>/`.
  - Signatur: `function [output] = compute_SSM(sys,n)`
- **`matGV2.m`** (89 Zeilen) — Liest die symbolische Nichtlinearität, identifiziert Monomial-Beiträge, baut für jede Ordnung $i$ die Sparse-Matrix $G_i \in \mathbb{R}^{2n \times (2n)^i}$ in der modalen Basis.
  - Signatur: `function varout = matGV2(f,x,T,order)`
  - Status-Message beim Start (Code-Statement, kein Kommentar): `hwait=waitbar(0,'Building up matrices');` (`matGV2.m:6`)
- **`measure_inv_autonomous.m`** (137 Zeilen) — Berechnet den Invarianz-Fehler $\delta_{\text{inv}}$.
  - Signatur: `function [error_avg] = measure_inv_autonomous(rho_0,rho_1,N,T)`

#### Hilfsfunktionen für Polynom-Algebra (Kronecker-basiert)
- **`kronproduct.m`** (12 Zeilen) — Iterierter Kronecker-Produkt-Wrapper: `kron(K_1, kron(K_2, ...))`.
- **`nkron.m`** (11 Zeilen) — $n$-faches Kronecker-Produkt einer einzigen Matrix mit sich selbst.
- **`kronGK1n.m`** (3 Zeilen) — $G_n \cdot K_1^{\otimes n}$ — der "reine" Beitrag der höchsten Nichtlinearität auf den linearen Modus.
- **`kronGK.m`** (23 Zeilen) — Symmetrisierte Summation von $G_m$ mit Kronecker-Produkten der niedrigerordigen $K_{?}$. Verwendet `nsumk(m,n)` um Multi-Indizes mit Summe $n$ und Länge $m$ zu enumerieren.
- **`kronKR.m`** (13 Zeilen) — Berechnet $\sum_m K_m \cdot \mathrm{Sym}(I^{\otimes ?} \otimes R_{?})$ — die DW·R-Beiträge für die cohomologische Gleichung.
- **`combinator.m`** (≥260 Zeilen) — Kombinatorik-Helfer (Permutationen mit/ohne Wiederholung, Kombinationen mit/ohne Wiederholung). Wird mit `combinator(2,i,'p','r')` aufgerufen, um alle Indextupel der Länge $i$ aus $\{1,2\}$ zu generieren.
- **`nsumk.m`** (27 Zeilen) — Liefert alle nicht-negativen $n$-Tupel, die zu $k$ summieren. Wird in `kronGK.m` und `check_res.m` benutzt.
  - Kommentar:
> "NSUMK Number and listing of non-negative integer n-tuples summing to k"
> — `SSMtool/SSMtool/nsumk.m:1`
- **`cumsumall.m`** + **`cumsumall.cpp`** + **`cumsumall.mexw32`** — MEX-File für schnelles cumulative-sum auf Integer-Arrays. Wird intern von `combinator.m` benutzt:
> "Keep in mind that the usefullness of this MEX-File is limited because of saturation for most problems outside of use with COMBINATOR."
> — `SSMtool/SSMtool/cumsumall.m:9-10`

#### Symbol-/String-Helfer
- **`str2sym.m`** (≈230 Zeilen) — Polyfill für ältere MATLAB-Versionen, die `str2sym` noch nicht haben.
- **`sym2cell.m`** (≈10 Zeilen) — Symbolische Matrix → Zell-Array von Strings.
- **`sym2char.m`** (47 Zeilen) — Symbolische oder numerische Matrix → MATLAB-String der Form `[a,b;c,d]`.

#### Integration
- **`int_dyn.m`** (30 Zeilen) — Integration des vollen Systems via `ode15s`.
- **`int_dyn_em.m`** (28 Zeilen) — Wie `int_dyn.m`, aber mit Interpolation auf gleichmäßiges Gitter (für Vergleich mit reduzierter Trajektorie). Das `_em` deutet auf "error-measurement".
- **`int_red_dyn.m`** (37 Zeilen) — Integration der reduzierten Polar-Dynamik $\dot\rho, \dot\theta$ via `ode45`.
- **`int_red_dyn_em.m`** (63 Zeilen) — Wie `int_red_dyn.m`, aber mit Event-Detection (Trajektorie stoppt, sobald sie den inneren Kreis $\rho_\epsilon$ erreicht).

#### Auto-generierte Funktions-Files (vom Tool selbst geschrieben)
- **`R_sub_function.m`** (8 Zeilen) — Beispiel-File: aktuelle reduzierte Dynamik $R(z)$ aus dem letzten `compute_SSM`-Lauf. Wird von `int_red_dyn.m` evaluiert.
- **`R_sub_EM_function.m`** (5 Zeilen) — analoge Variante für Error-Measurement.
- **`res_function.m`** (9 Zeilen) — Resonanz-Test-Funktion, geschrieben von `check_res.m`.
- **`system_function.m`** (auto-generiert, Größe variabel) — Aktuelles vollständiges Vektorfeld als symbolische Funktion, geschrieben von `measure_inv_autonomous.m:59-60` via `matlabFunction(dyn_sys, 'file', 'system_function', 'Vars', [spv]);`. Die Anzahl der Variablen entspricht `numel(spv)` und ist damit von der Phasenraum-Dimension des aktuell geladenen Systems abhängig (z.B. 4 für ein 2-DoF-Mech-System mit $x_1, x_2, \dot x_1, \dot x_2$).

#### Plot
- **`plot_SSM.m`** (128 Zeilen) — 3D-Surface-Plot der SSM in physikalischen Koordinaten. Sampled $\rho \in [0, r_{\max}], \theta \in [0,2\pi]$, evaluiert $W(\rho \cos\theta, \rho\sin\theta)$, plottet als `surfl`.

#### Status / Misc
- **`Spinner.m`** (62 Zeilen) — Java-basierter Busy-Spinner für das GUI.
- **`updatewaitbar.m`** (8 Zeilen) — Updatet Progress-Bar.
- **`cs.mat`** — Side-effect-File: speichert den aktuellen `folder_id` (Timestamp) zwischen Funktionsaufrufen, damit alle Helfer auf das gleiche `Data/run_<timestamp>/` zugreifen.
- **`cluster_info.mat`** — Cached `cpuNum` aus dem Parallel-Pool-Setup.
- **`Examples/2DOF_inner_res.mat`** — 2-DOF Shaw-Pierre-System mit near-inner-1:1-Resonanz.
- **`Examples/2DOF_outer_res.mat`** — 2-DOF Shaw-Pierre-System mit near-outer-Resonanz.
- **`Examples/beam.mat`** — 16-Element diskretisierter nichtlinearer Timoshenko-Beam (Phasenraum-Dimension 32).

### 5.2 `Addendum_Isolas/` — Isolas-Erweiterung (2018, kein GUI)

- **`startup.m`** — Adds `core/`, `example_beam/`, `example_SP/` zum MATLAB-Pfad.

#### `Addendum_Isolas/core/` — Wiederverwendbare Bausteine
- **`genlexd.m`** (22 Zeilen) — Generiert das nächste $n$-Tupel in absteigender lexikographischer Reihenfolge (für die Aufzählung von Polynomexponenten).
- **`nch.m`** (3 Zeilen) — `nchoosek(order+nvar-1, nvar-1)` — Anzahl der Polynome in `nvar` Variablen vom Grad `order`.
- **`man2cor.m`** (14 Zeilen) — Mappt eine SSM-Mannigfaltigkeit (gegeben als Cell-Array von Polynom-Tensoren) auf die physikalische $i$-te Koordinate via $\sum_i T(i,j) W_j$.
- **`man2cor_ab.m`** (20 Zeilen) — Wie `man2cor`, aber für die zwei Fourier-Komponenten ($a, b$) der zeit-periodischen $W_1$.
- **`poly_power.m`** (108 Zeilen) — Berechnet ordnungsweise $U^p$ (Polynom hoch $p$) via Rekurrenz $H_p(k) = (p/k_i)\sum_{m\le k, m_i>0}(m_i \cdot U(m) \cdot H_{p-1}(k-m))$.
  - Signatur: `function [H_CO,H_COij] = poly_power(U,Uij,Hp,Hpij,numvar,corder,power)`
  - Kommentar (`poly_power.m:6`):
> "% General recurrence formula: H_p(k) = (p/k_i)Sum_(m<=k, m_i>0)(m_i*U(m)H_(p-1)(k-m))"
- **`poly_product_DWR.m`** (108 Zeilen) — Berechnet $DW(z) R(z)$ ordnungsweise. Kernoperation der cohomologischen Gleichung.
  - Kommentar (`poly_product_DWR.m:1`):
> "% Comment: the 0th order can be computed with this function"
- **`poly_product_DW0R1.m`** — Berechnet $DW_0(z) R_1(z,\phi)$ für die nicht-autonome Korrektur erster Ordnung.
- **`poly_product_DW1R0.m`** — Berechnet $DW_1(z,\phi) R_0(z)$.
- **`poly_product_ab.m`** — Produkt zweier Polynome in der $(a,b)$-Fourier-Darstellung.

#### `Addendum_Isolas/example_SP/` — Modified Shaw-Pierre 2-DOF
- **`ex_SP_W0_3_W1_0.m`** (≥600 Zeilen) — Vollständiges Skript: 2-DOF-System, autonome SSM bis Ordnung 3, **ohne** non-autonome Korrektur ($W_1 = 0$, also nur autonomer Backbone).
- **`ex_SP_W0_5_W1_5.m`** (≈900 Zeilen) — Vollständiges Skript: SSM-Ordnung 5 für $W_0$ UND Ordnung 5 für $W_1$. Berechnet **forced response curves**, identifiziert Stabilität, plottet Isolas.
  - Forcing-Term im Code (`ex_SP_W0_5_W1_5.m:30`):
```matlab
fphi = [P*cos(omega_f*t);0];
```
  - System: 2 Massen, lineare und kubische Federn, kubische **und** quintische Nichtlinearitäten:
```matlab
fnl = [gamma_3*x3^3+gamma_5*x3^5+kappa*x1^3;0];
```

#### `Addendum_Isolas/example_beam/` — Bernoulli-Beam mit kubischer Endfeder
- **`L_Bernoulli_Beam_Model.m`** (53 Zeilen) — Generiert $M, C, K, f$ für einen Finite-Element-diskretisierten Euler-Bernoulli-Beam. Strukturelle Dämpfung $C = \alpha M + \beta K$. Nichtlinearität: kubische Feder am letzten Knoten. `n` = Anzahl FE-Elemente; Phasenraum-Dimension $4n-2$.
  - Kommentar (`L_Bernoulli_Beam_Model.m:2-7`, wörtlich, Zeilenumbrüche aus Original):
> ```
> %This code gives the Mass, Damping and Stiffness Matrix for a Forced linear
> %Bernoulli Beam with nonlinear cubic spring attached to the last node. It
> %also gives a forcing amplitude vector f. The beam is forced in terms of an
> %displacement imposed on the last node. The equations of motion are
> %M*u''+C*u'+K*u+f_nl=f. Structural damping with coefficients alpha and beta
> %is used for the damping matrix C.
> ```
- **`ex_beam_W0_3_W1_0.m`** (≈600 Zeilen) — Wie `ex_SP_W0_3_W1_0.m`, aber für den Beam.

---

## 6. Demos und Beispiele

### 6.1 GUI-Predefined-Examples (`SSMtool/SSMtool/Examples/`)

#### Example 1: `2DOF_inner_res.mat` — 2-DOF Shaw-Pierre near-INNER-resonance
- 2 Massen, 3 lineare Federn, kubische Nichtlinearität an Mass 1.
- Charakterisiert in Ponsioen et al. 2018 (J. Sound Vib.) Sec. 7.1.
- ODE-Dimension: 4 (Phasenraum), `n_dof = 2`.
- **Demonstriert**: SSM bei innerer 1:1-Resonanz; Backbone-Curve mit Modus-Kopplung.
- Manual Fig. 2.3 oben:
> "modified Shaw-Pierre example as explained in section 7.1 and 7.2 of Ponsioen et al. [2]"
> — `SSMtool_manual.pdf:8`

#### Example 2: `2DOF_outer_res.mat` — 2-DOF Shaw-Pierre near-OUTER-resonance
- Gleiches Modell, andere Parameter — externe Quasi-Resonanz zwischen Master und Rest.

#### Example 3: `beam.mat` — Diskretisierter nichtlinearer Timoshenko-Beam
- 32-dimensionaler Phasenraum (16 Knoten × 2 DOF × 2 für Geschwindigkeit).
- Charakterisiert in Ponsioen et al. 2018 Sec. 7.3.
- **Demonstriert**: SSMtool auf Systemen mit moderat hoher Dimension.

### 6.2 Addendum-Skripte (NICHT in GUI integriert)

#### `Addendum_Isolas/example_SP/ex_SP_W0_3_W1_0.m`
- 2-DOF Shaw-Pierre, $W_0$ bis Ordnung 3, **kein** $W_1$ — nur autonomer Backbone als Sanity-Check.
- Parameter (`ex_SP_W0_3_W1_0.m:11-22`): $m_1=m_2=1$, $c_1=0.03$, $c_2=0.03\sqrt{3}$, $k_1=k_2=3$, $\kappa=0.4$, $\gamma_3=-0.6$, $\gamma_5=0$.

#### `Addendum_Isolas/example_SP/ex_SP_W0_5_W1_5.m`
- Gleiches System, Ordnung 5 für $W_0$, Ordnung 5 für $W_1$, Forcing $\epsilon P\cos(\omega_f t)$.
- **Demonstriert**: Vollständige FRC- und Isola-Berechnung über die analytische Vorhersage (kein Continuation nötig).
- $\gamma_5 = 1.2$ (statt 0 wie in der einfacheren Variante) — quintische Nichtlinearität sorgt für Isolas.

#### `Addendum_Isolas/example_beam/ex_beam_W0_3_W1_0.m`
- Bernoulli-Beam mit kubischer Endfeder, FE-Diskretisierung.

### 6.3 Was im Repo NICHT vorhanden ist
- 3D-ODE-Beispiele (alle Demos sind mechanische 2-DOF oder höhere mit gerader Dimension).
- Slow/fast-Kopplungs-Demos.
- Nicht-mechanische ODE-Beispiele.
- SSMs der Dimension > 2 (V1.0 ist hardcoded auf 2D-SSMs).

---

## 7. Outputs

### 7.1 Was SSMtool nach `Compute SSM` ablegt

In `Data/run_<timestamp>/`:
- **`SSM_function_<timestamp>.m`** — auto-generierte MATLAB-Funktion `[q_1,q_2,...,q_{2n}] = SSM_function_<ts>(z1,z2)`. Das ist die Polynomdarstellung von $W(z_1,z_2)$ im gewählten Output-Koordinatensystem (modal/physical/complex).
- **`R_function_<timestamp>.m`** — auto-generierte Funktion `[zd_1,zd_2] = R_function_<ts>(z1,z2)`. Das ist die reduzierte Dynamik $R(z_1,z_2)$.
- **`sys_<timestamp>.mat`** — Snapshot der `sys`-Struktur ($M$, $C$, $K$, $f$, $\lambda$, $T$, etc.) für spätere Reproduktion.

### 7.2 GUI-Sichtbare Outputs
- **Polynom-Ausdrücke** für $W(z)$ und $R(z)$ als Text im "SSM Parameterization"-Fenster (Manual Fig. 2.7, S. 11).
- **3D-Surface-Plot** der SSM in physikalischen Koordinaten — projiziert auf 3 wählbare Koordinaten.
- **Reduced Dynamics in Polar-Form**: $\dot\rho = a(\rho)$, $\dot\theta = b(\rho)$.
- **Backbone-Curve**: Plot $|x_i|_{\max}$ vs. $\Omega = b(\rho)$, oder $\langle |x| \rangle$ vs. $\Omega$.
- **Trajektorien-Vergleich**: 3D-Plot der reduzierten und vollen Trajektorie auf der SSM (Manual Fig. 2.10).
- **Invarianz-Fehler**: skalare Zahl $\delta_{\text{inv}}$.

### 7.3 Outputs aus dem Addendum (Isolas)
- **`mech_sys_isola.m`**, **`mech_sys_isola_dx.m`**, **`mech_sys_isola_dp.m`** — auto-generierte Funktionen für die gemittelte (slow-flow) Dynamik und ihre Jacobi-Matrizen, geeignet als Input für Continuation-Tools wie COCO.
- **Konturplot** $G(\rho, \Omega) = 0$ in der $(\Omega, \rho)$-Ebene — zeigt FRC und Isolas.
- **Stabilitätsklassifikation** der Lösungen via Jacobi-Eigenwerte.

---

## 8. Limitationen (kritisch und vollständig)

### 8.1 Strukturelle Limitationen (V1.0)
1. **Nur mechanische Systeme**. Die Eingabe verlangt $M, C, K, f$. Generelle First-Order-ODEs müssen vom User in mechanische Form gebracht werden. Falls das nicht möglich ist (z. B. weil es keine kanonische Position-Velocity-Aufspaltung gibt), ist SSMtool nicht direkt anwendbar.
2. **Nur 2D-SSMs**. Hardcoded — `K1 = [eye(2);zeros(sys_dim-2,2)]` (`compute_SSM.m:47`) macht aus dem Master-Subraum genau 2 Spalten. Höher-dimensionale SSMs (z. B. 4D für Bursting-Modes) sind nicht berechenbar.
3. **Nur autonom**. Forcing wird im V1.0 NICHT unterstützt; das Addendum_Isolas ist eine separate Skript-Sammlung, kein integrierter GUI-Pfad.
4. **Fixpunkt am Ursprung erforderlich**. Wenn der echte Fixpunkt nicht am Ursprung liegt, muss der User vorher koordinatentransformieren.
5. **Nicht-trivialer Fixpunkt-Check ist Pflicht** — `compute_SSM.m:16-22` bricht ab.
6. **Asymptotische Stabilität ist Pflicht im nicht-konservativen Branch.** `compute_subspace.m:40-43` bricht ab, wenn ein Eigenwert nicht-negativen Realteil hat — aber **nur** wenn `conservative == false`. Im **konservativen Branch** (`compute_subspace.m:12-26`) werden imaginär-achsen-Eigenwerte explizit zugelassen und nach $|\mathrm{Im}\,\lambda|$ aufsteigend sortiert; die zugehörige Resonanz-Behandlung ist separat in `check_res.m:65-72` implementiert. Saddle-Knoten, Hopf-instabile Fixpunkte und Sattel-Fokus-Konfigurationen werden also nur im nicht-konservativen Standard-Pfad abgelehnt.
7. **$A$ muss semisimpel sein** (`compute_subspace.m:8-10`). Jordan-Blöcke werden abgelehnt.
8. **Master-Subraum muss aus genau 2 Eigenwerten bestehen**: entweder ein komplex-konjugiertes Paar oder zwei reelle Eigenwerte, aber nicht ein reeller plus die Hälfte eines komplexen Paares (`orderT.m:81-87`).
9. **Maximum SSM-Ordnung ist 50** (`SSM.m:575-579`).

### 8.2 Numerische und praktische Limitationen
10. **Symbolische Taylor-Entwicklung** der Nichtlinearität via `taylor()` skaliert schlecht mit $n$ und Ordnung. Für $n \approx 30$ und Ordnung $\ge 5$ wird das Symbolic-Toolbox-Setup zum Bottleneck.
11. **Speicherbedarf**: Die Polynomdarstellung in `compute_SSM.m` benutzt redundante $2^k$-Indizierung statt der minimalen $\binom{k+1}{1}$-Indizierung. Für Ordnung 10 und 2 Variablen sind das 1024 Koeffizienten pro Mode und Ordnung statt 11.
12. **Externe Resonanz-Toleranz $10^{-4}$** ist hardcoded. Bei knapp resonanten Systemen kann das zu falsch-positiven Abbrüchen führen.
13. **Innere Resonanz-Toleranz $5\cdot 10^{-2}$** ist relativ großzügig. Bei "almost resonant" Modi werden viele Terme als resonant deklariert und in $R$ verschoben.
14. **Lokalität**: SSM ist eine **formale Reihe** um den Fixpunkt. Konvergenzradius ist im allgemeinen unbekannt. In der Praxis funktioniert die Reihe nur in einer Umgebung, deren Größe der Spektralgap und die Glattheit von $f$ bestimmen. Für lightly-damped Mech-Systeme typischerweise einige Prozent der maximalen Amplitude.
15. **Parallel-Toolbox-API-Abhängigkeit**: Das Manual sagt explizit, dass paralleles Rechnen **nicht erforderlich** ist. Der Code ruft aber dennoch direkt `gcp`, `parallel.defaultClusterProfile`, `parcluster`, `parpool` (`SSM.m:674-686`) und `spmd` (`compute_SSM.m:141, 157-163`) auf. Korrekte Formulierung: **mehrere Worker sind nicht nötig**, aber die Parallel-Computing-Toolbox muss installiert sein, weil diese API-Aufrufe sonst fehlschlagen. Ohne installierte Toolbox crasht der Aufruf, auch wenn ein einzelner Worker für die Rechnung ausreichen würde.
16. **Dateimanagement-Side-effects**: SSMtool schreibt `cs.mat`, `cluster_info.mat`, `Data/run_<ts>/`, `R_sub_function.m`, `R_sub_EM_function.m`, `res_function.m`, `system_function.m` ins MATLAB-Working-Directory. Jeder neue Lauf überschreibt diese Files — kein sauberes Multi-Tenant-Verhalten.

### 8.3 Was die Pipeline NICHT prüft
- Konvergenz der Reihe in einer **konkreten** Umgebung — nur die Existenz von Termen wird garantiert.
- Globale Eigenschaften der SSM (sie ist nur lokal definiert).
- Robustheit gegenüber Parameter-Störungen oder Modellfehlern.

---

## 9. Anwendbarkeit auf das LPPL-System des Users

### 9.1 Das System (`/home/hz/Data/LPPL-forced/LPPL-attractor/lpplattr02_ode.py`)

Der relevante autonome Kern (ohne Halvings, ohne Sign-OU, ohne Damping) ist (`lpplattr02_ode.py:75-90`):

```python
def system_of_equations(y, noisy_values, t_current=1, ...):
    y1, y2, z = y
    dy1 = y2 - Z_A * y2 * z + Z_MIX * (y1 - y2)
    dy2 = (noisy_values['alpha'] * y2 * abs(y2)**(noisy_values['M'] - 1)
           - noisy_values['gamma'] * y1 * abs(y1)**(noisy_values['N'] - 1)
           + Z_B * y1 * z)
    if use_damping and DAMPING["enabled"]:
        ...
        dy2 += (-2.0 * kappa * y2) - (stiff * y1)
    dz = -Z_C * z + Z_D * y1 + Z_E * y1 * y2
    return np.array([dy1, dy2, dz])
```

In Symbolen mit Parametern aus `lpplattr02_params.py`:
$$\dot y_1 = y_2 - Z_A\,y_2 z + Z_{\text{MIX}}(y_1-y_2)$$
$$\dot y_2 = \alpha\, y_2 |y_2|^{M-1} - \gamma\, y_1 |y_1|^{N-1} + Z_B\, y_1 z + (\text{optional Damping-Terme})$$
$$\dot z = -Z_C\, z + Z_D\, y_1 + Z_E\, y_1 y_2$$

mit $\alpha = -7.4\cdot 10^{-4}$, $\gamma = 0.003$, $M \approx 1.071$, $N=3$, $Z_A=Z_B=8\cdot 10^{-3}$, $Z_C=0.0039$, $Z_D=10^{-6}$, $Z_E=2$, $Z_{\text{MIX}}=2\cdot 10^{-4}$.

### 9.2 Probleme bei direkter Anwendung von SSMtool

#### Problem 1: Nicht-mechanische Struktur
Das System ist 3D im Phasenraum, NICHT in $(y,\dot y)$-Form mit gerader Dimension. SSMtool erwartet eine `M`/`C`/`K`-Aufspaltung. **Es gibt keine kanonische mechanische Form** für dieses LPPL-System, weil $z$ keine Geschwindigkeit ist, sondern eine separate slow-feedback-Variable.
- **Konsequenz**: Der User kann SSMtool V1.0 NICHT direkt benutzen, ohne den Code zu modifizieren.
- **Workaround**: Den V1.0-Eingangscode umgehen und direkt in das Innere von `compute_SSM.m` einsteigen — dort wird intern alles auf First-order $\dot x = Ax + F_{nl}(x)$ transformiert. Aber: Die GUI führt diese Transformation hardcoded über `M, C, K` aus. Eine "mechanische Hülle" $M=I, C=0, K=0$ würde funktionieren, wenn man $y_1 = $ Position, $y_2 = $ Geschwindigkeit interpretiert — aber dann fehlt $z$ als dritte Komponente.
- **Bessere Option**: Skripte aus `Addendum_Isolas/core/` direkt benutzen (`poly_product_DWR.m`, `poly_power.m`, `man2cor.m`, `genlexd.m`, `nch.m`) — die sind allgemein und nicht an mechanische Struktur gebunden. Der User muss dann den Wrapper-Code analog zu `ex_SP_W0_3_W1_0.m` selbst schreiben, mit `ndof_spv = 3` statt `4`, und die Nichtlinearität als symbolische Polynome aufstellen.

#### Problem 2: Power-Law-Nichtlinearität ist NICHT analytisch (Theorie-Annahme aus CFdlL, kein Code-Check)
**Wichtig zur Einordnung:** SSMtool prüft Analytizität / Glattheit der Nichtlinearität **nirgends explizit im Code**. Die Pipeline ruft `taylor()` auf den symbolischen Ausdruck (`compute_SSM.m:24-35`) und nimmt damit implizit an, dass die Nichtlinearität eine konvergente Taylor-Reihe um den Fixpunkt hat. Die Forderung nach Analytizität (bzw. ausreichender $C^r$-Glattheit) ist eine **Voraussetzung des Cabré-Fontich-de la Llave-Theorems** für die Existenz und Eindeutigkeit der SSM, nicht etwas, das SSMtool intern checken würde. Wenn die Annahme verletzt ist, läuft SSMtool eventuell trotzdem durch — aber das Ergebnis hat keine theoretische Garantie.

Für das LPPL-System bedeutet das:
- $|y_1|^{N-1}$ mit $N=3$ ist okay: $y_1\cdot |y_1|^2 = y_1^3$ für reelles $y_1$, also analytisch (sogar polynomiell).
- $|y_2|^{M-1}$ mit $M = 1.071$: $|y_2|^{0.071}$. Das ist **nicht analytisch** am Ursprung — es ist nicht einmal differenzierbar an $y_2 = 0$. **Theoretische Konsequenz**: das Cabré-Fontich-de la Llave-Theorem ist auf das LPPL-System mit nicht-ganzzahligem $M$ strikt nicht anwendbar; SSMs in Haller'schem Sinne sind formal nicht garantiert. SSMtool selbst wird das nicht melden — es würde versuchen, eine Taylor-Entwicklung durchzuführen, die in dieser Form mathematisch nicht existiert.
- Selbst wenn man $M=1$ exakt setzt (sodass $y_2|y_2|^0 = y_2$ — das ist linear, kein nichtlinearer Beitrag), hat man immer noch $\gamma y_1^3$ als kubische Nichtlinearität, und das System wird wieder analytisch.

**Konsequenz**: Mit dem aktuellen Power-Law $M=1.071$ ist die SSM-Theorie nach Cabré-Fontich-de la Llave strikt nicht anwendbar. SSMtool würde es trotzdem versuchen, aber das Ergebnis wäre nicht theoretisch fundiert. Der User müsste entweder:
- $M$ auf eine ganze Zahl runden (wie in Standard-Duffing-/Van-der-Pol-Modellen, z.B. $M=1$ oder $M=3$), oder
- die Power-Law durch ein **anderes glattes Ersatzmodell** ersetzen, z.B. eine Sigmoid-Approximation $y_2 (\epsilon^2 + y_2^2)^{(M-1)/2}$ mit $\epsilon > 0$, das am Ursprung analytisch ist und für $|y_2| \gg \epsilon$ in das ursprüngliche Power-Law übergeht. Eine direkte Taylor-Entwicklung von $y_2|y_2|^{M-1}$ am Ursprung **gibt es nicht**, weil die Funktion dort nicht differenzierbar ist — eine Regularisierung ist Pflicht.

#### Problem 3: Fixpunkt-Lokalisierung
Der triviale Fixpunkt $(y_1, y_2, z) = (0,0,0)$ ist ein Fixpunkt:
- $\dot y_1(0) = 0 - 0 + 0 = 0$ ✓
- $\dot y_2(0) = 0 - 0 + 0 = 0$ ✓ (weil $y_2 |y_2|^{M-1} \to 0$ für $y_2\to 0$ wenn $M>1$, ja)
- $\dot z(0) = 0$ ✓

**ABER**: Die Linearisierung am Ursprung fehlt im quadratischen Anteil von $y_2|y_2|^{M-1}$ — die Ableitung dieser Funktion bei $y_2=0$ ist
$$\frac{d}{dy_2}\bigl(y_2 |y_2|^{M-1}\bigr)\bigg|_{y_2=0} = M\cdot |y_2|^{M-1}\big|_{y_2=0} = \begin{cases}0 & M>1 \\ 1 & M=1 \\ \infty & M<1\end{cases}$$
Also ist die Ableitung 0 für $M=1.071$ — der Term ist lokal flach. Linearisierung ist trotzdem definiert: nur $\gamma y_1 |y_1|^{N-1} = \gamma y_1^3$ gibt ebenfalls keinen linearen Beitrag (nur kubisch), und der lineare Anteil wird allein von den Z-Termen und dem $y_2$-Term in $\dot y_1$ getragen.

#### Problem 4: Linearisierung am Ursprung (Jacobi $A$)
Streng linearisiert wäre:
$$A = \begin{pmatrix} \partial_{y_1}\dot y_1 & \partial_{y_2}\dot y_1 & \partial_z \dot y_1 \\ \partial_{y_1}\dot y_2 & \partial_{y_2}\dot y_2 & \partial_z \dot y_2 \\ \partial_{y_1}\dot z & \partial_{y_2}\dot z & \partial_z \dot z \end{pmatrix}\bigg|_{(0,0,0)} = \begin{pmatrix} Z_{\text{MIX}} & 1 - Z_{\text{MIX}} & 0 \\ 0 & 0 & 0 \\ Z_D & 0 & -Z_C \end{pmatrix}$$
(weil $|y_2|^{M-1}\big|_0 = 0$ für $M>1$, $|y_1|^{N-1}\big|_0 = 0$ für $N>1$, und $Z_E y_1 y_2$ linearisiert sich zu null am Ursprung).

Mit Werten: $Z_{\text{MIX}} = 2\cdot 10^{-4}$, $Z_C = 0.0039$, $Z_D = 10^{-6}$:
$$A \approx \begin{pmatrix} 2\cdot 10^{-4} & 0.9998 & 0 \\ 0 & 0 & 0 \\ 10^{-6} & 0 & -0.0039 \end{pmatrix}$$

Die Eigenwerte sind:
- Charakteristisches Polynom: $\det(A - \lambda I) = 0$.
- Block-Struktur: zweiter Spalte/Reihe ist null in Zeile 2 — also ist $\lambda = 0$ ein Eigenwert (mit Eigenvektor $(0,1,0)^T$ entlang $y_2$).
- Die übrigen zwei Eigenwerte aus dem $(y_1, z)$-Block:
$$A_{2\times 2} = \begin{pmatrix} Z_{\text{MIX}} & 0 \\ Z_D & -Z_C \end{pmatrix}$$
mit Eigenwerten $\lambda = Z_{\text{MIX}} = 2\cdot 10^{-4}$ und $\lambda = -Z_C = -0.0039$.

**Drei Eigenwerte**: $\lambda_1 = 0$ (entartet), $\lambda_2 = 2\cdot 10^{-4} > 0$ (instabil), $\lambda_3 = -3.9\cdot 10^{-3}$ (stabil).

**Das ist keine asymptotische Stabilität**. Zwei der drei Eigenwerte haben nicht-negativen Realteil. SSMtool würde mit `'Real part for each eigenvalue of Spec(A) must be strictly negative'` abbrechen.

Der **dynamisch interessante** Fall im LPPL-System ist eine Hopf-artige Instabilität des Ursprungs mit $y_1, y_2$ als oszillierende Variablen — d. h. das System hat ein **stable limit cycle** mit nichttrivialer Frequenz, NICHT einen asymptotisch stabilen Ursprung. SSMtool ist für genau diesen Fall **nicht entworfen**.

#### Problem 5: Halvings = Zeit-Abhängigkeit = Autonomie-Bruch
Der `kappa_and_stiffness`-Term in `lpplattr02_ode.py:32-36` und der Halving-Mechanismus in `get_cycle_number`/`get_mu_t`/`get_sigma_t` (`lpplattr02_ode.py:39-67`) sind explizit zeitabhängig:
```python
def kappa_and_stiffness(t_current, M2_val, kappa0, t_min):
    t = max(float(t_current), float(t_min))
    kappa     = kappa0 * (M2_val / t)
    ...
```
- Die Damping-Stärke skaliert mit $1/t$ — das ist ein klassischer non-autonomer LPPL-Term.
- Halvings ändern $\mu$ und $\sigma$ in Stufen — das ist ein PWM-artiges, zeitabhängiges Forcing.
- Die Sign-OU-Komponente (`sign_step_substeps`) ist sogar stochastisch.

**Konsequenz**: SSMtool V1.0 ist autonom-only. Selbst das Addendum_Isolas behandelt nur **periodisches** Forcing (nicht $1/t$, nicht stufenweise PWM). Das LPPL-System verletzt die Autonomie-Voraussetzung in fundamentaler Weise.

### 9.3 Welche SSM-Dimension wäre theoretisch sinnvoll?
Wenn man die Power-Law-Nichtlinearitäten durch Polynom-Approximationen ersetzt, das System um einen wirklich stabilen Fixpunkt linearisiert (z. B. nach Modifikation der Z-Parameter, sodass alle Eigenwerte negativen Realteil haben), und die Zeit-Abhängigkeiten ignoriert, dann wäre eine **2D-SSM** über dem $(y_1, y_2)$-Oszillations-Modus die natürliche Wahl: $z$ ist die slow-feedback-Variable und sollte nicht im Master-Subraum sein. Das passt zur SSMtool-V1.0-Beschränkung auf 2D-SSMs.

### 9.4 Bewertung
| Kriterium | Status | Schweregrad |
|---|---|---|
| Mechanische $M, C, K$-Form | Nicht vorhanden | Hard-Block für GUI-Pfad; lösbar via Eigenbau-Skripte aus `Addendum_Isolas/core/` |
| Analytizität von $f$ | Verletzt durch $|y_2|^{M-1}$ mit $M$ nicht-ganzzahlig | Hard-Block; nur lösbar durch Approximation |
| Asymptotische Stabilität am Ursprung | NICHT erfüllt mit aktuellen Parametern (zwei Eigenwerte $\ge 0$) | Hard-Block; eventuell durch Parameter-Verschiebung lösbar, aber dann ist es ein anderes System |
| Autonomie | Verletzt durch Halvings, Damping $\propto 1/t$, Sign-OU | Hard-Block für V1.0; teilweise lösbar mit Addendum_Isolas, aber nur für **periodisches** Forcing |
| 2D-SSM-Geometrie | Sinnvoll wenn $(y_1,y_2)$ als Master und $z$ als slow-feedback | Konzeptionell OK |
| 3D-Phasenraum | Akzeptabel wenn man `ndof_spv = 3` nimmt und Eigenbau-Skripte schreibt | Lösbar mit manueller Arbeit |
| Power-Laws $M$, $N$ ganzzahlig? | $N=3$ ist OK, $M=1.071$ nicht | Hard-Block oder Approximation |

**Gesamturteil**: Das LPPL-System verletzt **mehrere unabhängige** Voraussetzungen von SSMtool. Eine direkte Anwendung ist NICHT möglich. Teilweise Anwendbarkeit setzt voraus, dass der User ein vereinfachtes, autonomes Modell mit ganzzahligen Power-Laws und einem stabilen Fixpunkt formuliert. Selbst dann ist nicht klar, ob das vereinfachte Modell die für den User interessante Dynamik (die LPPL-Power-Law-Singularität, das slow drift in $z$, die halvings) noch widerspiegelt.

---

## 10. Konkrete Checkliste: Was der User MATLAB-seitig tun müsste

Falls der User SSMtool dennoch ausprobieren will (z. B. auf einem stark vereinfachten Modell):

### 10.1 Vorbereitung
1. MATLAB R2016b oder neuer installieren (Manual S. 5).
2. **Symbolic Math Toolbox** installieren (zwingend für `taylor`, `sym`, `eig` symbolisch).
3. **Parallel Computing Toolbox** installieren (zwingend, weil `compute_SSM.m:160-163` `spmd` benutzt).
4. SSMtool aus `/home/hz/Data/Attractor/SSMtool/SSMtool/` ins MATLAB-Working-Directory laden.

### 10.2 System aufbereiten
5. Power-Law-Nichtlinearitäten ersetzen:
   - $y_2 |y_2|^{M-1}$ mit $M=1.071$: ersetzen durch $y_2$ (linear) oder durch $y_2^3$ (kubisch, aber falsche Skalierung) oder durch eine Polynom-Approximation um den Arbeitspunkt. **Genaue Wahl muss der User physikalisch motivieren.**
   - $\gamma y_1 |y_1|^{N-1}$ mit $N=3$: das ist $\gamma y_1^3$, OK als Polynom.
6. Zeit-Abhängigkeiten entfernen:
   - Halvings ausschalten (`HALVING_ENABLED=[False]*10` oder explizit ignorieren).
   - Damping ausschalten oder konstant halten (statt $\propto 1/t$).
   - Sign-OU ignorieren (deterministische Approximation).
7. Fixpunkt explizit lokalisieren: $(0,0,0)$ ist trivial, aber prüfen, ob er für die gewählten Parameter wirklich asymptotisch stabil ist (Eigenwerte $A$ alle mit negativem Realteil).
8. Falls **nicht** stabil: Parameter so verschieben, dass alle drei Eigenwerte negativen Realteil haben — z. B. $Z_{\text{MIX}}$ negativ machen, eine echte Damping-Konstante hinzufügen.

### 10.3 SSMtool-Pfad: NICHT der GUI-Pfad
Weil das LPPL-System keine kanonische $M\ddot y + C\dot y + Ky$-Struktur hat, ist der GUI-Pfad NICHT brauchbar. Stattdessen:

#### Option A: Eigenbau via `Addendum_Isolas/core/`
9. Schreibe ein eigenes Top-Level-Skript analog zu `ex_SP_W0_3_W1_0.m` mit `ndof_spv = 3` (statt 4) und `nvar = 2` (Master-Subraum-Dimension).
10. Setze die symbolische Nichtlinearität direkt:
```matlab
syms y1 y2 z real
f = [-Z_A*y2*z; Z_B*y1*z + Z_E*y1*y2 - gamma*y1^3; Z_E*y1*y2];
```
(nach Vereinfachungen aus Schritt 5).
11. Bilde $A$ analytisch durch Linearisierung: `A = double(jacobian(f + [y2; 0; 0], [y1;y2;z]))` an $(0,0,0)$.
12. Eigendekomposition: `[X,D] = eig(A); lambda = diag(D);`
13. Wähle 2 Eigenwerte für den Master (idealerweise das komplex-konjugierte Paar mit kleinstem $|\mathrm{Re}\lambda|$).
14. Iteriere mit den Polynom-Helfern (`poly_product_DWR`, `poly_power`, `man2cor`, `nch`, `genlexd`) ordnungsweise wie in `ex_SP_W0_3_W1_0.m:131-269`.
15. Cohomologische Gleichung manuell aufstellen: pro Mode $j$ und Polynomindex $(m_1,m_2)$:
$$W_j(m) = \frac{\mathrm{RHS}_j(m) - \mathrm{Resonanz}}{\lambda_j - m_1\lambda_1 - m_2\lambda_2}$$

#### Option B: GUI-Pfad mit künstlichem mechanischem Wrapper (NICHT empfohlen)
16. Definiere eine künstliche $M, C, K, f$-Struktur mit $n=2$ DOF, sodass die ersten beiden Variablen $(y_1, y_2)$ die Master-Mode bilden. $z$ wird dann zur "Geschwindigkeit" der zweiten DOF gemacht — ABER das verletzt die Bedeutung von $z$ als slow-feedback. Man verliert die Interpretation, und die Eigenwerte des resultierenden $A$ entsprechen nicht den tatsächlichen Eigenwerten des LPPL-Systems.
17. **Fazit zu Option B**: Funktioniert nur mit massivem Verlust an Bedeutung. Nicht empfohlen.

### 10.4 Was der User NICHT erreichen wird
- Eine SSM-Beschreibung des **echten** LPPL-Systems mit allen Halvings, allen Power-Laws, allem Sign-OU. Das übersteigt die theoretischen Voraussetzungen von SSMtool und CFdlL fundamental.
- Aussagen zu globalen Strukturen wie Heteroklinen-Verbindungen, Limit-Cycles abseits des Ursprungs, oder dem Übergangsverhalten zwischen Halvings.

### 10.5 Was der User stattdessen evaluieren sollte
- **SSMLearn** (data-driven): nimmt Trajektorien aus dem LPPL-Simulator und versucht, eine SSM aus den Daten zu lernen. Verlangt nicht die theoretischen Voraussetzungen von SSMtool, dafür aber genügend gute Daten.
- **Center-Manifold-Berechnung** für das slow-fast-System mit $z$ als slow-Variable — das ist konzeptionell näher an der LPPL-Struktur als SSM.
- **Averaging / Slow-Manifold-Theory** für den $\dot z = -Z_C z + \ldots$-Anteil mit $1/Z_C \approx 256$ als langsamer Zeitskala, gegenüber der schnellen $(y_1,y_2)$-Oszillation.

---

## 11. Zusammenfassung der wichtigsten Code-Pfade (Cheat-Sheet für Codex)

| Schritt | Datei | Funktion | Zeilen |
|---|---|---|---|
| GUI-Start | `SSM.m` | `SSM_OpeningFcn` | 48-117 |
| System laden | `SSM.m` | `push_loadmech_Callback` | 1128-1207 |
| Linearisierung | `compute_subspace.m` | `compute_subspace` | 1-49 |
| Stabilitäts-Check | `compute_subspace.m` | inline | 40-44 |
| Master-Subraum | `SSM.m` | `push_select_lambda_Callback` | 482-539 |
| Basis-Reorder | `orderT.m` | `orderT` | 1-90 |
| Resonanz-Test | `check_res.m` | `check_res` | 1-75 |
| Höher-Order Resonanzen | `check_higher_res.m` | `check_higher_res` | 1-90 |
| Compute SSM main | `compute_SSM.m` | `compute_SSM` | 1-224 |
| Trivial-FP-Check | `compute_SSM.m` | inline | 16-22 |
| Taylor von $f$ | `compute_SSM.m` | inline | 24-37 |
| $W_1$, $R_1$ Init | `compute_SSM.m` | inline | 44-49 |
| $G_m$ aufbauen | `matGV2.m` | `matGV2` | 1-89 |
| Innere-Resonanz-Lokalisierung | `compute_SSM.m` | inline | 72-95 |
| Cohomologische Schleife | `compute_SSM.m` | inline | 100-169 |
| $K_R$ Beitrag | `kronKR.m` | `kronKR` | 1-13 |
| $G_K$ Beitrag | `kronGK.m` | `kronGK` | 1-23 |
| $G_{K_1}$ Beitrag | `kronGK1n.m` | `kronGK1n` | 1-3 |
| Lösung pro Ordnung | `compute_SSM.m` | `spmd`-Block | 156-165 |
| Output-Schreiben | `compute_SSM.m` | inline | 208-218 |
| Backbone | `SSM_exp.m` | `push_backbone_Callback` | 207-300 |
| Polar-Form | `SSM_exp.m` | inline | 141-142 |
| Trajektorien voll | `int_dyn.m` | `int_dyn` | 1-30 |
| Trajektorien reduziert | `int_red_dyn.m` | `int_red_dyn` | 1-37 |
| Invarianz-Fehler | `measure_inv_autonomous.m` | `measure_inv_autonomous` | 1-137 |
| **Addendum (Forced)** | | | |
| 2-DOF SP, $W_0=3$, $W_1=0$ | `Addendum_Isolas/example_SP/ex_SP_W0_3_W1_0.m` | gesamt | 1-≈600 |
| 2-DOF SP, $W_0=5$, $W_1=5$ | `Addendum_Isolas/example_SP/ex_SP_W0_5_W1_5.m` | gesamt | 1-≈900 |
| Beam, $W_0=3$, $W_1=0$ | `Addendum_Isolas/example_beam/ex_beam_W0_3_W1_0.m` | gesamt | 1-≈600 |
| FRC-Implizite-Gleichung | `ex_SP_W0_5_W1_5.m` | inline | 666-674 |
| Isola-Konturplot | `ex_SP_W0_5_W1_5.m` | inline | 730-770 |
| Polynom-Helfer (allgemein) | `Addendum_Isolas/core/poly_product_DWR.m` | `poly_product_DWR` | 1-108 |
| Polynom-Helfer (allgemein) | `Addendum_Isolas/core/poly_power.m` | `poly_power` | 1-108 |
| Polynom-Helfer (allgemein) | `Addendum_Isolas/core/man2cor.m` | `man2cor` | 1-14 |
| Polynom-Helfer (allgemein) | `Addendum_Isolas/core/genlexd.m` | `genlexd` | 1-22 |
| Polynom-Helfer (allgemein) | `Addendum_Isolas/core/nch.m` | `nch` | 1-3 |

---

## 12. Wörtliche Schlüssel-Zitate aus dem Manual (für Quellen-Validierung)

> "$\dot{z} = R(z)$"
> — `SSMtool_manual.pdf:4` (Gleichung 1.2)

> "$\dot{\rho} = a(\rho), \quad \Omega = \dot{\theta} = b(\rho)$"
> — `SSMtool_manual.pdf:4` (Gleichung 1.3)

(abridged, Auslassungen explizit markiert)
> "After specifying an autonomous mechanical system, press `Analyze` to extract the eigenvalues of the system […Zwischensätze ausgelassen…] Select a two-dimensional spectral subspace $\mathcal{E}$, by selecting a pair of complex conjugate eigenvalues and pressing the `Select` button […Zwischensätze ausgelassen…] SSMtool will calculate the spectral quotient $\sigma(\mathcal{E})$, which indicates the minimum order of the Taylor expansion needed to be able to the capture the unique SSM. Additionally, SSMtool will check if the outer non-resonance conditions […genaue Bedingungs-Formel ausgelassen…] are satisfied in order to guarantee that the SSM of interest exists […Zwischensätze ausgelassen…] In case of a lightly damped spectral subspace $\mathcal{E}$, SSMtool will check for near inner-resonances […Detailtext ausgelassen…] up to the order dictated by the spectral quotient."
> — `SSMtool_manual.pdf:8`

> "You are now required to specify the order of the Taylor expansion and select if the output coordinates of the SSM should be in modal, physical or complex coordinates. The 'Check for higher-order internal resonances' checkbox is there to specify if SSMtool should check for near internal-resonances up to the order of expansion instead of the order dictated by the spectral quotient"
> — `SSMtool_manual.pdf:9`

> "$\delta_{\text{inv}} = \frac{1}{N}\sum_{i=1}^N \frac{\mathrm{dist}(i)}{\max_{\theta\in S^1}\|\tilde{\mathbf{x}}(\rho_0,\theta)\|_2}, \quad \mathrm{dist}(i) = \max\left\|\mathbf{x}_i\big|^{\rho_\epsilon}_{\rho_0} - \tilde{\mathbf{x}}_i\big|^{\rho_\epsilon}_{\rho_0}\right\|_2$"
> — `SSMtool_manual.pdf:14` (Gleichung 2.1)

> "We have improved the core of SSMtool to handle systems with time-periodic forcing. We have used the exact reduced dynamics on two-dimensional time-periodic spectral submanifolds (SSMs) to extract forced-response curves (FRCs) and predict isolas in arbitrary multi-degree-of-freedom mechanical systems without performing costly numerical simulations [3]."
> — `README.md:20-22`

---

## 13. Referenzen (aus Manual, S. 15)

1. G. Haller and S. Ponsioen. *Nonlinear normal modes and spectral submanifolds: existence, uniqueness and use in model reduction*. Nonlinear Dyn., 86(3):1493–1534, 2016.
2. S. Ponsioen, T. Pedergnana, and G. Haller. *Automated computation of autonomous spectral submanifolds for nonlinear modal analysis*. Journal of Sound and Vibration, 420:269–295, 2018.
3. S. Ponsioen, T. Pedergnana, and G. Haller. *Analytic Prediction of Isolated Forced Response Curves from Spectral Submanifolds*. arXiv preprint arXiv:1812.06664, 2018.
4. T. Breunung and G. Haller. *Explicit backbone curves from spectral submanifolds of forced-damped nonlinear mechanical systems*. Proc. R. Soc. A, 474(2213):20180083, 2018.
5. S. Jain, P. Tiso, and G. Haller. *Exact nonlinear model reduction for a von Kármán beam: slow-fast decomposition and spectral submanifolds*. J. Sound Vib., 423:195–211, 2018.

Theoretischer Hintergrund (CFdlL):
- X. Cabré, E. Fontich, R. de la Llave. *The parameterization method for invariant manifolds*. Indiana Univ. Math. J. 52 (2003).

---

## 14. Was an SSMtool ungewöhnlich oder unterspezifiziert ist

1. **Hardcoded 2D-Master-Dimension**: Die Variable `nvar = 2` ist überall im Code fest verankert. Eine Verallgemeinerung auf 4D, 6D etc. erfordert nicht-triviale Code-Änderungen in `compute_SSM.m`, `kronKR.m`, `kronGK.m`.
2. **Redundante $2^k$-Polynombasis** statt $\binom{k+1}{1}$ — verschwendet Speicher, aber vereinfacht die Kronecker-Produkt-Algebra.
3. **`spmd`-Pflicht**: SSMtool versucht implizit, einen Parallel-Pool zu starten. Wenn das fehlschlägt (z. B. ohne Parallel Toolbox), bricht der Lauf bei `cluster = load('cluster_info.mat')` ab, weil die Datei noch nicht existiert.
4. **Side-effects ins Working-Directory**: SSMtool legt mehrere Dateien im aktuellen MATLAB-Pfad ab (`cs.mat`, `cluster_info.mat`, auto-generierte `.m`-Files). Mehrere parallele SSMtool-Instanzen würden sich gegenseitig überschreiben.
5. **`SSM.m` mischt GUI-Code (GUIDE-generiert) mit Numerik-Aufrufen** — unsauber, aber lauffähig.
6. **Manual ist sehr knapp** (15 Seiten) und bezieht sich für die Mathematik fast vollständig auf das JSV-2018-Paper [2]. Dieses Paper ist die eigentliche Referenz für alle Formeln; das Manual ist nur ein Bedienungs-Tutorial.
7. **Die "complex coordinates"-Option** in `SSM.m:594-597` ist nur teilweise dokumentiert — sie liefert die SSM in den komplexen Master-Koordinaten ohne Rück-Transformation.
8. **Numerische Stabilität** bei Ordnung > 10 ist nicht garantiert; das `round(...,17)` in `compute_SSM.m:177-178` ist ein primitiver Filter.
9. **Keine Unit-Tests, keine Validierungs-Suite** im Repo. Die Korrektheit wird allein durch die Predefined-Examples und die Veröffentlichung in J. Sound Vib. validiert.

---

## Anhang A: Funktions-Index (im Workflow-Text erwähnte `.m`-Dateien)

Alphabetisch sortiert. Diese Liste enthält NUR die Funktionen, die im Workflow-Text dieser MD referenziert werden — sie ist keine vollständige Inventur des Repos (`SSMtool/SSMtool/` enthält insgesamt ~189 `function`-Definitionen, die meisten als GUI-Callbacks und triviale Helfer).

| Funktion | Pfad | Zweck |
|----------|------|-------|
| `check_higher_res.m` | `SSMtool/SSMtool/` | Resonanz-Check über die SSM-Polynom-Ordnung hinaus, Toleranz $5\cdot 10^{-2}$ |
| `check_res.m` | `SSMtool/SSMtool/` | Externe + interne Resonanz-Check vor Compute_SSM, Toleranzen $10^{-4}$ / $5\cdot 10^{-2}$ |
| `combinator.m` | `SSMtool/SSMtool/` | Kombinatorik-Helfer (Permutationen/Kombinationen mit/ohne Wiederholung) |
| `compute_SSM.m` | `SSMtool/SSMtool/` | **Hauptfunktion**: löst die cohomologischen Gleichungen ordnungsweise, schreibt SSM- und R-Funktionen nach `Data/run_<ts>/` |
| `compute_subspace.m` | `SSMtool/SSMtool/` | Eigendecomposition + Master-Subraum-Auswahl + Stabilitäts-Abbruch (`~conservative`-Branch) |
| `cumsumall.m` + `.cpp` + `.mexw32` | `SSMtool/SSMtool/` | MEX-File für schnelle cumulative-sum auf Integer-Arrays (intern in `combinator.m`) |
| `ex_beam_W0_3_W1_0.m` | `SSMtool/Addendum_Isolas/example_beam/` | Bernoulli-Beam-Beispiel: Forced Response Order $\le 3$ + Isolas |
| `ex_SP_W0_3_W1_0.m` | `SSMtool/Addendum_Isolas/example_shawpierre/` | Shaw-Pierre Forced Response, Polynom-Ordnung 3 |
| `ex_SP_W0_5_W1_5.m` | `SSMtool/Addendum_Isolas/example_shawpierre/` | Shaw-Pierre Forced Response, Polynom-Ordnung 5 |
| `genlexd.m` | `SSMtool/Addendum_Isolas/core/` | Lexikographische Generierung von Multi-Index-Tupeln |
| `int_dyn.m` | `SSMtool/SSMtool/` | Integration des vollen Systems via `ode15s` |
| `int_dyn_em.m` | `SSMtool/SSMtool/` | Wie `int_dyn.m`, mit Interpolation auf gleichmäßiges Gitter (Error-Measurement) |
| `int_red_dyn.m` | `SSMtool/SSMtool/` | Integration der reduzierten Polar-Dynamik $(\dot\rho, \dot\theta)$ via `ode45` |
| `int_red_dyn_em.m` | `SSMtool/SSMtool/` | Wie `int_red_dyn.m`, mit Event-Detection auf $\rho_\epsilon$ |
| `kronGK.m` | `SSMtool/SSMtool/` | Symmetrisierte Summation $G_m$ mit Kronecker-Produkten der niedrigerordigen $K$ |
| `kronGK1n.m` | `SSMtool/SSMtool/` | $G_n \cdot K_1^{\otimes n}$ — reiner Beitrag der höchsten Nichtlinearität |
| `kronKR.m` | `SSMtool/SSMtool/` | $\sum_m K_m \cdot \mathrm{Sym}(I^{\otimes ?}\otimes R_?)$ — DW·R-Beiträge der cohomologischen Gleichung |
| `kronproduct.m` | `SSMtool/SSMtool/` | Iterierter Kronecker-Produkt-Wrapper |
| `L_Bernoulli_Beam_Model.m` | `SSMtool/Addendum_Isolas/example_beam/` | Generiert $M, C, K, f$ für FE-diskretisierten Euler-Bernoulli-Beam mit kubischer Endfeder |
| `man2cor.m` | `SSMtool/Addendum_Isolas/core/` | Konvertiert Mannigfaltigkeits-Koeffizienten in physikalische Koordinaten |
| `man2cor_ab.m` | `SSMtool/Addendum_Isolas/core/` | $a$-$b$-Variante von `man2cor.m` |
| `matGV2.m` | `SSMtool/SSMtool/` | Liest die symbolische Nichtlinearität, baut für jede Ordnung die Sparse-Matrix $G_i \in \mathbb{R}^{2n \times (2n)^i}$ in modaler Basis |
| `measure_inv_autonomous.m` | `SSMtool/SSMtool/` | Berechnet den Invarianz-Fehler $\delta_{\text{inv}}$; erzeugt `system_function.m` via `matlabFunction` |
| `mech_sys_isola.m` | `SSMtool/Addendum_Isolas/core/` | Mechanisches System für Isolas-Detektion |
| `mech_sys_isola_dx.m` | `SSMtool/Addendum_Isolas/core/` | Räumliche Ableitung für Isolas-Continuation |
| `mech_sys_isola_dp.m` | `SSMtool/Addendum_Isolas/core/` | Parameter-Ableitung für Isolas-Continuation |
| `nch.m` | `SSMtool/Addendum_Isolas/core/` | $n$-choose-$k$-Helfer |
| `nkron.m` | `SSMtool/SSMtool/` | $n$-faches Kronecker-Produkt einer einzigen Matrix mit sich selbst |
| `nsumk.m` | `SSMtool/SSMtool/` | Liefert alle nicht-negativen $n$-Tupel, die zu $k$ summieren |
| `orderT.m` | `SSMtool/SSMtool/` | Ordnung der modalen Transformation $T$; erzwingt 2D-Master-Subraum (reell oder cc-Paar) |
| `plot_SSM.m` | `SSMtool/SSMtool/` | 3D-Surface-Plot der SSM in physikalischen Koordinaten |
| `poly_power.m` | `SSMtool/Addendum_Isolas/core/` | Symbolische Polynom-Potenzen |
| `poly_product_ab.m` | `SSMtool/Addendum_Isolas/core/` | Produkt zweier symbolischer Polynome ($a$-$b$-Variante) |
| `poly_product_DW0R1.m` | `SSMtool/Addendum_Isolas/core/` | $DW_0 \cdot R_1$-Produkt für Forced-Order-1-Term der Invarianz-Gleichung |
| `poly_product_DW1R0.m` | `SSMtool/Addendum_Isolas/core/` | $DW_1 \cdot R_0$-Produkt für Forced-Order-1-Term |
| `poly_product_DWR.m` | `SSMtool/Addendum_Isolas/core/` | Allgemeines Polynom-Produkt $DW \cdot R$ |
| `R_sub_function.m` | `SSMtool/SSMtool/` | Auto-generiert: aktuelle reduzierte Dynamik $R(z)$ aus letztem `compute_SSM`-Lauf |
| `R_sub_EM_function.m` | `SSMtool/SSMtool/` | Auto-generiert: $R$-Variante für Error-Measurement |
| `res_function.m` | `SSMtool/SSMtool/` | Auto-generiert von `check_res.m`: Resonanz-Test-Funktion |
| `Spinner.m` | `SSMtool/SSMtool/` | Java-basierter Busy-Spinner für GUI |
| `SSM.m` | `SSMtool/SSMtool/` | GUI-Hauptfenster, ruft `gcp/parcluster/parpool` für Parallel-Pool |
| `SSM_int.m` | `SSMtool/SSMtool/` | GUI-Callback für Integration |
| `SSM_exp.m` | `SSMtool/SSMtool/` | GUI-Callback für Export |
| `SSM_invar.m` | `SSMtool/SSMtool/` | GUI-Callback für Invarianz-Plot |
| `startup.m` | `SSMtool/Addendum_Isolas/` | Pfad-Initialisierung für Addendum_Isolas |
| `str2sym.m` | `SSMtool/SSMtool/` | Polyfill für ältere MATLAB-Versionen |
| `sym2cell.m` | `SSMtool/SSMtool/` | Symbolische Matrix → Zell-Array von Strings |
| `sym2char.m` | `SSMtool/SSMtool/` | Symbolische/numerische Matrix → MATLAB-String `[a,b;c,d]` |
| `system_function.m` | `SSMtool/SSMtool/` | Auto-generiert von `measure_inv_autonomous.m:59-60` via `matlabFunction(dyn_sys, 'file', 'system_function', 'Vars', [spv])`; Variablenanzahl = `numel(spv)` |
| `updatewaitbar.m` | `SSMtool/SSMtool/` | Updatet Progress-Bar |

---

Ende der Dokumentation.
