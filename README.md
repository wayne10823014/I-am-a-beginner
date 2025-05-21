## 一、**摘要（Abstract）**
隨著 GPU 加速在生物資訊領域日漸普及，但傳統 CUDA／OpenCL 解決方案必須安裝驅動並受限於特定硬體，對線上教學與臨床前端分析造成不便。2024 年正式標準化的 **WebGPU** 透過單一 JavaScript API 對接 Vulkan／D3D12／Metal，兼具「免安裝、跨硬體、資料留在本機」三大優勢。本研究以高強度 **Pair-Hidden Markov Model Forward (Pair-HMM Forward)** 演算法為例，評估 WebGPU 的效能與可行性。

我們以周育晨（2024）公開之 C++／CUDA 程式為基準，首先撰寫 WebGPU Baseline，接著針對 CPU↔GPU 往返、BindGroup 重建與全域記憶體延遲等瓶頸，依序導入「單一 CommandBuffer 批次提交」「Dynamic Uniform Offset」及「Workgroup Cache」，形成 **WebGPU-Optimized**。在 NVIDIA RTX 2070 Super、Apple M1 與 Intel UHD 620 測試長度 10²–10⁵ 的序列後，Optimized 相較 Baseline 呈現 **顯著加速（最高逾百倍）**，執行速度可達 **CUDA 的八成以上**；三款裝置的 Log-Likelihood 相對誤差均低於 10⁻⁵，且在無 NVIDIA GPU 時對單執行緒 C++ 亦提供多達數十至數百倍的加速。

本研究證實僅憑 JavaScript + WGSL，即能於瀏覽器中於秒級完成 Pair-HMM Forward 計算，並提出三項瀏覽器端專屬優化策略及跨硬體實測結果。**此成果為 Web-native 基因體分析工具的普及奠定基礎，推動生物資訊運算的民主化與即時化。**

# 第 1 章　緒論（Introduction）

## 1.1 研究背景（Background）

高通量定序（Next-Generation Sequencing, NGS）技術的蓬勃發展，使得基因體資料的規模與複雜度以指數速度攀升，生物資訊分析對計算效能提出前所未有的挑戰。**Pair Hidden Markov Model（Pair-HMM）Forward 演算法**能同時支援序列比對、基因型鑑定與變異偵測，是眾多基因體流程不可或缺的運算核心。

然而現行分析流程（如 GATK、Samtools 及 BWA-MEM2）大多以 C++ 或 Python 實作，再藉 NVIDIA CUDA 或 OpenCL 取得 GPU 加速。使用者除了安裝驅動、設定 SDK 與相依函式庫，還受限特定 GPU 架構；教育現場或資源受限的實驗室，常因無法配置高階 GPU 或雲端計算額度，而被迫採用 CPU 模式，導致運算成本與等待時間倍增。雲端服務雖能降低本機安裝複雜度，卻伴隨帳號管理、網路延遲與敏感資料外流的顧慮。

自 2024 年 5 月 WebGPU 正式進入 Chrome 穩定通道以來，Firefox Nightly 與 Edge Dev 亦陸續提供實驗支援。WebGPU 以單一 JavaScript API 映射 Vulkan、Direct3D 12 與 Metal，並運行於瀏覽器沙盒，讓 NVIDIA、AMD、Intel 乃至 Apple Silicon GPU 皆能「免安裝驅動」即時執行平行運算。因此，WebGPU 被視為打破硬體與作業系統藩籬的嶄新契機，有望大幅降低生物資訊工具的使用門檻，特別是在教學、臨床前端及低資源環境。

## 1.2 研究動機與目的（Purpose）

雖然 WebGPU 具備「免安裝、跨硬體、資料留在本機」三大優勢，但其設計初衷仍以圖形與機器學習推論為主，高強度動態規劃演算法面臨：

* **API 排程開銷**：Pair-HMM Forward 需順序處理對角線，若每次 dispatch 都產生 CommandBuffer 與 BindGroup，CPU↔GPU 往返將急速累積。
* **缺乏全域同步**：演算法每一條對角線都依賴前一條結果；WebGPU 僅提供 workgroup 級同步，無法如 CUDA 以 `__syncthreads()`+ kernel-return 階段性同步整塊資料。
* **缺少專用特殊函式單元（SFU）**：Pair-HMM 頻繁呼叫 `log`/`exp`; CUDA 的 SFU 可 4 cycle 完成 `log₂`, 而 WebGPU 需以 ALU+LUT+FMA 近似，多出 2–4 倍延遲。
* **高記憶體存取需求**：DP 矩陣為可讀寫 storage buffer，在 WebGPU 預設「跳過 L1」的情況下，頻繁全域讀寫導致 DRAM 往返居高不下。

上述特性放大了 WebGPU 尚未完善的 API 與硬體限制，學界與業界對其處理生物資訊高強度工作負載的可行性仍缺乏系統性驗證。故本研究聚焦：「在瀏覽器環境下，WebGPU 能否以足夠效能執行 Pair-HMM Forward，並作為 CUDA 的實用替代？」。

## 1.3 研究方法與主要結果（Methods & Results）

本研究以周育晨（2024）公開之 C++／CUDA 程式為基準對照，採以下 **三項專屬優化策略** 以對應前述瓶頸：

1. **單一 CommandBuffer 批次提交**：將多次對角線運算收攏至一次 `queue.submit()`，消除大量 API 往返延遲。
2. **Dynamic Uniform Offset**：將靜態參數置於 uniform buffer，以動態偏移 (dynamic offset) 避免重建 BindGroup。
3. **Workgroup Cache**：將 emission／transition 常量顯式搬入 `var<workgroup>`，降低跨對角線反覆讀取全域記憶體的成本。

在 NVIDIA RTX 2070 Super 上針對序列長度 100–100 000 測試，**WebGPU-Optimized** 相較 Baseline 加速 6.8–142×，並達到 CUDA 11–84 % 的效能；在 Apple M1 與 Intel UHD 620 上亦對純 CPU 提供 4–463× 加速（序列長度 ≥ 1 000），**Log-Likelihood 誤差低於 10⁻⁵**。

## 1.4 結論與貢獻（Conclusion）

本研究證明：只需 JavaScript 與 WGSL，即能在瀏覽器沙盒中於秒級時間完成中大型 Pair-HMM Forward 計算。三項互補之瀏覽器端優化策略有效消弭 WebGPU 的 API、同步與記憶體三大瓶頸；跨 NVIDIA、Apple、Intel 硬體的驗證顯示方法具「廠商不可知性」。此成果為「打開瀏覽器即用」的基因體分析鋪路，亦為未來 **雙精度支援 與 WASM-SIMD + WebGPU 混合加速** 奠定實證基礎。


## 第 2 章 文獻探討
### 2.1 生物資訊中的高性能計算需求
#### 2.1.1 高通量定序與計算挑戰
隨著高通量定序（Next-Generation Sequencing, NGS）技術的快速發展，基因體資料的規模與複雜度呈指數增長。根據 Illumina 官方規格，NovaSeq X Plus 系統使用 25B flow cell 於雙流道運行可產生約 52 billion（520 億）條 reads（2 × 150 bp），完整運行時間約 48 小時，換算為每小時約 3.25 × 10¹¹ 鹼基對（Illumina, 2024）。此類分析涉及大量序列比對與概率計算，對計算資源的需求遠超傳統 CPU 架構所能負擔。例如，序列比對工具如 BWA（Li & Durbin, 2010）和 Bowtie（Langmead et al., 2009）需處理數百萬至數十億條短序列，理論上最壞情況時間複雜度達 O(NM)（N、M 為序列長度），但實務上因採用 FM-index 進行種子篩選與擴張，平均複雜度近似線性 O(L)（L 為讀取長度）。這些挑戰促使研究者尋求高效能計算（High-Performance Computing, HPC）解決方案，尤其是 GPU 加速技術，以滿足生物資訊分析的即時性需求。
#### 2.1.2 Pair-HMM Forward 演算法的核心角色
Pair-Hidden Markov Model（Pair-HMM）Forward 演算法是生物資訊中序列比對與基因型鑑定的核心組件。該演算法基於隱馬可夫模型（HMM），透過動態規劃計算兩序列間的對齊概率，廣泛應用於工具如 GATK（McKenna et al., 2010）和 Samtools（Li et al., 2009）。根據 Durbin et al. (1998, 第 4 章）及 Banerjee et al. (2017），Pair-HMM Forward 的時間複雜度為 O(NM)，其中 N、M 分別為參考序列與讀取序列的長度，空間複雜度可降至 O(max(N,M))。對於長序列（N ≈ 10⁴），此演算法構成顯著計算瓶頸。近年研究顯示，透過 GPU 平行化可顯著加速此過程，例如 Schmidt et al. (2024) 的 CUDA 實作在 NVIDIA RTX 4090 上將 32 × 12 kb 片段的計算時間從數小時縮減至分鐘級（少於 3 分鐘）。然而，該演算法對記憶體存取模式與計算精度要求高，需針對硬體特性進行優化。
### 2.2 傳統 GPU 加速方案：CUDA 與 OpenCL
#### 2.2.1 CUDA 在生物資訊的應用
NVIDIA CUDA 作為 GPU 加速的主流框架，已在生物資訊領域取得顯著成功。例如，Liu et al. (2013) 開發了 CUDASW++ 3.0，利用 CUDA 加速 Smith-Waterman 演算法，實現比單執行緒 CPU 快 30–90 倍的序列比對。Schmidt et al. (2024) 進一步將 CUDA 應用於 Pair-HMM Forward，展示其在高通量基因型鑑定中的高效能。然而，CUDA 的應用受限於 NVIDIA 專屬硬體，且需安裝驅動程式與 CUDA Toolkit，對非專業使用者構成門檻。此外，CUDA 程式需針對特定 GPU 架構（如 Ampere、Hopper）優化，跨硬體相容性較差。
#### 2.2.2 OpenCL 的跨平台嘗試
OpenCL 旨在提供跨硬體的 GPU 加速解決方案，支援 NVIDIA、AMD 與 Intel GPU。根據 Stone et al. (2010)，OpenCL 在科學計算中展現潛力，例如加速分子動力學模擬。然而，其在生物資訊的應用較 CUDA 有限，主要因硬體支援不均與程式開發複雜度高。例如，Klöckner et al. (2012) 指出 OpenCL 的記憶體管理與執行緒同步機制在不同硬體間表現差異顯著，導致效能不穩定。此外，OpenCL 生態系統相較 CUDA 較不成熟，缺乏廣泛的函式庫支援，限制其在生物資訊中的普及。
#### 2.2.3 傳統方案的門檻與局限
總結而言，CUDA 與 OpenCL 雖在效能上具優勢，但均需繁瑣的前置設定（如驅動安裝、SDK 配置），對教育現場與臨床前端應用形成障礙。雲端 GPU 服務（如 AWS、Google Cloud）試圖解決本機設定問題，但引入資料傳輸延遲與隱私風險（Krampis et al., 2012）。此外，這些方案高度依賴特定硬體架構，無法實現真正的跨平台相容性，限制其在資源受限環境（如筆記型電腦或嵌入式設備）中的應用。
### 2.3 WebGPU 的興起與技術特性
#### 2.3.1 WebGPU 的技術背景
WebGPU 於 2024 年 12 月 19 日進入 W3C Candidate Recommendation Snapshot 階段，尚未成為正式 Recommendation，尚待完整實作與互通性測試（W3C, 2024）。WebGPU 透過單一 JavaScript API 映射 Vulkan、Direct3D 12 與 Metal 後端，支援 NVIDIA、AMD、Intel 及 Apple Silicon GPU。其核心優勢在於免安裝驅動、跨平台相容與瀏覽器沙盒的安全性，允許使用者僅需開啟瀏覽器即可執行平行運算。WebGPU 的計算管線（Compute Shader）以 WGSL（WebGPU Shading Language）撰寫，支援高效能矩陣運算與記憶體管理，適用於高強度運算任務。

![WebGPU 架構對應示意](https://hackmd.io/_uploads/H1GKrzs-ee.png)


#### 2.3.2 WebGPU 在高性能計算的潛力
WebGPU 在圖形渲染與機器學習領域已展現潛力。例如，MDN Web Docs（2025）介紹 WebGPU 在遊戲渲染中的高效能表現，實現媲美原生 Vulkan 的效果；TensorFlow.js 的 WebGPU 後端則加速了瀏覽器端神經網路訓練（TensorFlow.js Team, 2024）。Google Chrome Team (2024) 指出，Transformers.js 在 WebGPU 模式下於 NVIDIA RTX 4060 Laptop 測試 BERT-base 模型時，比 WebAssembly 快 32.51 倍，顯示其高效能潛力。這些案例表明 WebGPU 的計算模型能有效利用 GPU 的平行化能力。然而，其在生物資訊的應用尚處於初步階段，主要因該領域對計算精度與記憶體存取效率要求更高。WebGPU 的 Compute Shader 與傳統 GPU 框架的異同在於其抽象層設計，雖犧牲部分低階控制，但換取跨硬體的通用性。
#### 2.3.3 WebGPU 的挑戰與限制
儘管 WebGPU 具備跨平台優勢，其在高強度運算中仍面臨挑戰。首先，瀏覽器端的 API 排程開銷較高，尤其在頻繁的 CPU-GPU 資料傳輸中（Google Chrome Team, 2024）。其次，WebGPU 僅提供 workgroupBarrier，缺乏跨 workgroup 的全域同步機制，限制其在需要複雜執行緒協調的演算法中的表現。此外，瀏覽器沙盒限制了記憶體分配與特殊函式單元（SFU）的使用，WGSL 因未內建超越函數（transcendentals，如 log/exp）而依賴軟體模擬，雖底層 GPU 可能以 SFU 執行，但仍增加計算開銷；對雙精度浮點運算（f64）的支援亦不如 CUDA 完善（Jones, 2023）。Pair-HMM Forward 的波前（wavefront）依賴性要求頻繁的記憶體存取與執行緒同步，與 WebGPU 缺乏全域同步的特性形成挑戰，增加了實現高效能平行化的難度。
### 2.4 WebGPU 在生物資訊的初步探索
目前文獻中，WebGPU 在生物資訊的直接應用案例較少，但相關技術如 WebGL 與 WebAssembly（WASM）提供參考。例如，Ghosh et al. (2018) 利用 WebGL 實現 Web3DMol，展示瀏覽器端分子結構視覺化的可行性，證明 JavaScript 在生物資訊的可行性。WASM-SIMD（Single Instruction, Multiple Data）進一步提升了瀏覽器端運算效能，Jones (2023) 指出其與 WebGPU 的結合有望加速序列處理。此外，WebGPU 在其他高強度運算領域的應用提供了技術啟發。例如，TensorFlow.js 的 WebGPU 後端通過高效矩陣運算實現了瀏覽器端神經網路訓練的加速（TensorFlow.js Team, 2024），其對記憶體管理和並行化的優化策略為本研究移植 Pair-HMM Forward 提供了借鑑，特別是在處理高頻記憶體存取與計算密集任務時。 然而，這些研究多聚焦於視覺化或輕量計算，對於 Pair-HMM Forward 等高強度演算法的實作仍屬空白。Schmidt et al. (2024) 的 CUDA 實作為 WebGPU 移植提供了基準，但未探討瀏覽器端優化策略。
### 2.5 研究缺口與本研究的定位
綜合上述文獻，現有研究在以下方面存在不足：首先，WebGPU 在高強度生物資訊運算（如 Pair-HMM Forward）的效能與可行性尚未被系統性驗證；其次，針對瀏覽器端 GPU 計算的瓶頸（如 CPU-GPU 往返、BindGroup 重建、全域記憶體延遲），缺乏專屬優化策略，例如，WebGPU 的 setBindGroup() 呼叫涉及 V8-Blink-Dawn-Driver 多層驗證，單次延遲約 5–15 µs，對波前演算法的多次調用構成顯著開銷；最後，跨硬體（NVIDIA、Apple、Intel）的實測數據不足，無法充分評估 WebGPU 的通用性。例如，GATK HaplotypeCaller 等主流生物資訊工具依賴 CUDA 加速，需安裝 NVIDIA 專屬驅動與 Toolkit，限制了其在教育現場或資源受限環境（如無高階 GPU 的筆記型電腦）的部署；雲端方案則因資料傳輸與隱私問題難以滿足臨床前端需求。相較之下，WebGPU 的免安裝與跨硬體特性可大幅降低門檻，實現本地高效能運算。 本研究透過移植 周育晨 (2024) 的 CUDA 程式至 WebGPU，提出三項瀏覽器端優化策略—單一 CommandBuffer 批次提交、Dynamic Uniform Offset 與 Workgroup Cache—並於多硬體平台驗證其效能與精度。此成果不僅填補 WebGPU 在生物資訊應用的研究空白，亦為「免安裝、跨硬體、資料不離端」的基因體分析工具奠定基礎。


### 第 3 章　研究方法（Methods）

#### 3.1　數學模型（Mathematical Model）

本研究採用 **Pair-HMM 結合序列 Profile**。隱狀態為 Match（$M$）、Insert（$I$）、Delete（$D$），字母表 $\{A,C,G,T,-\}$。讀序列以機率矩陣

$$
\mathbf{P}=\bigl[p_{i,a}\bigr],\qquad
1\le i\le m,\; a\in\{A,C,G,T\},\;
\sum_{a} p_{i,a}=1,
$$

表示第 $i$ 個讀段為字元 $a$ 之機率；雜合序列為確定串 $h_1,\dots,h_n$。

轉移機率

$$
t_{XY},\qquad X,Y\in\{M,I,D\},
$$

與發射基礎矩陣 $\varepsilon_{X}(x,y)$ 皆沿用周育晨（2024）公開之 C++／CUDA 程式論文設定。

##### Profile 發射機率

$$
e^{M}_{i,j}= \sum_{a} p_{i,a}\,\varepsilon_{M}(a,h_j),
\qquad
e^{I}_{i,j}= \sum_{a} p_{i,a}\,\varepsilon_{I}(a,-),
$$

而 Delete 狀態讀端為 gap，故發射機率固定為 1。

---

#### 3.2　Pair-HMM Forward 演算法

1. **初始化**

$$
M_{0,0}=1,\; I_{0,0}=D_{0,0}=0,\quad
M_{0,j}=I_{0,j}=0,\;
D_{0,j}=\tfrac{1}{n}\quad(j>0).
$$

2. **遞迴**

$$
\begin{aligned}
M_{i,j}&=
 e^{M}_{i,j}\Bigl(
 t_{MM}M_{i-1,j-1}+t_{IM}I_{i-1,j-1}+t_{DM}D_{i-1,j-1}\Bigr),\\
I_{i,j}&=
 e^{I}_{i,j}\Bigl(
 t_{MI}M_{i-1,j}+t_{II}I_{i-1,j}\Bigr),\\
D_{i,j}&=
 t_{MD}M_{i,j-1}+t_{DD}D_{i,j-1}.
\end{aligned}
$$

3. **終端**

$$
P=\sum_{j=1}^{n}\!\bigl(M_{m,j}+I_{m,j}\bigr).
$$

4. **時間複雜度** 為 $\mathcal{O}(mn)$。

---

#### 3.2 系統設計與實作（System Design and Implementation）

##### 3.2.1 C++／CUDA 版

在前人工作已證實 CUDA 能以反對角線平行化高效實現 Pair-HMM 後，本研究**接手並調整周育晨（2024）公開之 C++／CUDA 程式**，將其原始雙精度 `double` 改為單精度 `float`，以建立與 WebGPU 版本可對等比較的「效能上限」對照組。由於 WebGPU 目前僅保證 f32 運算，若 CUDA 仍維持 f64，跨平台結果將被精度差異稀釋。經將單精度輸出與原始雙精度比對，長度 $N=10^{5}$ 時最大相對誤差僅 $2.18\times10^{-1}\%$，足以滿足後續 WebGPU 對照的精度門檻。首先，我們沿用「每條反對角線觸發一次 kernel」的結構；此設計雖需發射 \$2N\$ 次 kernel，但可藉由 `cudaDeviceSynchronize()` 在相鄰反對角線間形成 GPU-wide barrier，確保跨 thread-block 之資料相依完全滿足。如此一來，矩陣遞迴公式中左、上、左上的依賴便能以最直觀的方式映射至裝置端。

然而，僅有全域同步仍不足以隱藏記憶體延遲，因此在 block 內部我們維持 `__syncthreads()` 的微同步，讓每 32 threads 共享快取中的前一行結果後再進入下一輪計算。與此同時，對三條動態規劃陣列 \$M, I, D\$，我們採用主機端的「四行指標輪替」技巧，亦即固定配置四個長度 \(n+1\) 的緩衝區，並在 host 迴圈透過指標交換完成 `prev → curr → new` 的遞補；由於 CUDA 指標可被當作一般 C 指標操控，此做法免除了重新配置或 memcpy 的成本，最大化 PCIe／NVLink 帶寬的有效利用率。此結構同時為後續 WebGPU 移植鋪路：一旦進入瀏覽器環境失去可變指標，我們便需以 BindGroup 重建或 Dynamic Offset 取而代之。

![4. 全域同步差異圖](https://hackmd.io/_uploads/r19f4fsWeg.png)
圖 3-5　CUDA 與 WebGPU 在全域同步上的差異
CUDA 可於裝置端 kernel-return 插入 GPU-wide barrier；WebGPU 必須回到 JavaScript 重新 dispatch 才能觸發全域同步。

### 3.2.2 WebGPU Baseline

#### (一) 從 CUDA「多次 Kernel」到 WebGPU「多次 dispatch」

Pair-HMM Forward 的計算沿著動態規畫矩陣的反對角線 (wavefront) 逐步推進。每條 wavefront 必須等前一條全部完成後才能繼續。CUDA 最直觀的作法是在主機程式裡用 for 迴圈連續啟動 Kernel, 並在兩次 Kernel 之間呼叫 cudaDeviceSynchronize()。這相當於在 GPU 端插入全域 barrier, 同時讓主機程式得以安全交換三條 DP 指標。為確保計算正確性，Baseline 採用單一 compute pass 模擬 CUDA 的全域同步，雖犧牲部分效能，但簡化了移植流程。
然而移植到 WebGPU 時, WGSL 只有 workgroupBarrier(), 並無跨 workgroup 的同步原語, Shader 也無法像 CUDA Dynamic Parallelism 那樣在裝置端再啟動子工作負載。於是 **每一條反對角線都得由 JavaScript 端重新發起一次 dispatchWorkgroups()**。主機呼叫 queue.submit() 送出上一條 wavefront 後, 還得等待 device.queue.onSubmittedWorkDone() 才能更新 uniform 並提交下一個 dispatch。對長度 N 的序列而言, 這代表整體需要 **2N 次 dispatch, 也就有 2N 次 CPU↔GPU 往返**, 同步延遲完全暴露在 JavaScript 執行緒, 形成 Baseline 的第一個瓶頸。

#### (二) 指標輪替與 BindGroup 的不可變性

CUDA 只需在兩條 wavefront 之間交換三個 float\* 指標, 就能把 prev → curr → new 的角色依序推移, 驅動層無須重新配置資源。相對地, WebGPU 的 Buffer 綁定點在 **BindGroup 建立時即被鎖定**。若想讓下一條 wavefront 讀取新的 DP Buffer, 就只能重新呼叫 device.createBindGroup(), 把相同 binding slot 指向新的 GPUBuffer。這一過程會歷經 V8 → Blink → Dawn → Driver 等層層驗證, 單次耗時約 10–50 µs; 在 2N 次迴圈裡反覆執行, 累計延遲動輒數十秒, 因而出現第二個瓶頸。

![3. WebGPU 多層 IPC／驗證路徑](https://hackmd.io/_uploads/SJi1Nfi-gl.png)
圖 3-4　WebGPU Storage Buffer 綁定的多層 IPC / 驗證路徑與 CUDA 直接 Host→VRAM 相比，WebGPU 需歷經 JavaScript 執行緒→GPU Process→Driver IPC→VRAM 多段複製或驗證，導致單次 createBindGroup 延遲 5–15 µs。

#### (三) Shared memory 的缺席與高延遲 storage buffer 存取

在 CUDA 實作中, 九個 Transition 係數與七十五個 Emission 係數可預先載入 48 KB shared memory, 之後所有執行緒以約 80 ns 的延遲重複存取。雖然 WGSL 也支援 var<workgroup>, 但容量有限且必須由 Shader 手動搬運。Baseline 為求驗證正確性, 乾脆把小矩陣保留在 storage buffer。結果是每格計算需重複進行 6–9 次 global read, 單次延遲約 300 ns, 遠高於 shared memory, 這成了第三個瓶頸。

#### (四) Baseline 的暫行折衷

面對上述限制, Baseline 採取三項折衷措施。首先, 以「一次 compute pass 對應一條 wavefront」的方法, 用 GPU 指令天然序列化效果替代 cudaDeviceSynchronize(), 確保計算順序。其次, 在每條 wavefront 重新建立 BindGroup, 以顯式切換三顆 DP Buffer 的角色, 雖有 API 開銷, 卻能保證 Shader 讀寫方向正確。最後, 為避免額外負擔, Baseline 只在初始化階段建立一支 ComputePipeline, 全程重用同一 ComputePassEncoder, 至少省下重複編譯 WGSL 與重新產生 Pipeline State 的成本。

#### (五) Baseline 效能概況

在 NVIDIA RTX 2070 Super 上, 當序列長度 N = 100 000 時, Baseline 版本耗時 466 秒, 較同卡上的 CUDA 實作慢約兩個數量級。深入分析可知, 延遲主要來自「2N 次 IPC 同步」「2N 次 BindGroup 生成」以及「頻繁 storage buffer 讀取」。此結果清楚揭示 WebGPU 與 CUDA 架構差異的三個瓶頸, 也界定了下一節將提出的三項優化策略: **減少 Host↔GPU 往返、最小化 BindGroup 變動, 以及將熱點資料搬入 var<workgroup> 快取**。


### 3.2.3 WebGPU 優化版（本研究）

為真正解決 Baseline 在 (1) 頻繁主機同步、(2) 重複 BindGroup 建構、(3) 高延遲 storage 存取這三項瓶頸，我們依序導入「單一 CommandBuffer 批次提交」、「Dynamic Uniform Offset」與「Workgroup Cache」三種瀏覽器端優化策略。以下先簡介 WebGPU 的指令錄製與提交機制，隨後逐一說明各策略的設計動機、實作細節與實驗成效。

---

#### 3.2.3.1 單一 CommandBuffer 批次提交──降低 CPU–GPU 往返

**CommandEncoder 與指令流。** 在 WebGPU 中，`CommandEncoder` 負責錄製 `beginComputePass`、`dispatchWorkgroups`、`copyBufferToBuffer`、`end` 等命令。當呼叫 `encoder.finish()` 產生 `GPUCommandBuffer` 並以 `device.queue.submit([commandBuffer])` 提交後，GPU 會依序執行所有已錄製指令而無需 CPU 介入。只有在程式刻意呼叫 `await device.queue.onSubmittedWorkDone()` 時，JavaScript 執行緒才會同步等待整條指令流完成。

**傳統多次提交的痛點。** Baseline 為了維持對角線依賴，使用 `for` 迴圈逐條建構 encoder、`submit`、`await`，再動態重建下一條 encoder。對長度 N 的序列將產生 2N 次 「submit–barrier–encoder」 流程。每次等待不僅引發 CPU–GPU IPC 與瀏覽器排程延遲，也迫使 JavaScript 執行緒反覆進入 idle/active 狀態，整體開銷顯著。

**一次性指令流的優勢。** 我們保留對角線分段的程式邏輯，但僅在錄製階段多次 `beginComputePass`，最後只 submit 一次。如此 GPU 得以從頭到尾連續執行，CPU 無須插入同步；驅動驗證與排程成本大幅下降；`dispatch` 與 `copy` 命令連續排列也使 DRAM 流量更平穩。實驗顯示，把 2N 次 IPC 壓縮為 1 次後，總執行時間可明顯縮短。

![1. 單一 CommandBuffer 示意圖](https://hackmd.io/_uploads/BkGH7Mo-ee.png)
圖 3-6　單一 CommandBuffer 批次提交流程
主機（Host）先於指令錄製階段將多個 dispatch 收攏至同一 CommandBuffer，隨後一次性 submit 至 GPU 執行，省去 2 N 次 IPC 與排程開銷。
    
---

#### 3.2.3.2 Dynamic Uniform Offset──減少常量更新成本

**多小 buffer 與重複 BindGroup 問題。** Baseline 在每條對角線都需重建 BindGroup 以切換 dpPrev、dpCurr、dpNew 與各類常量 buffer，單次 `createBindGroup` 必經 V8 → Blink → Dawn 多層驗證，耗時 5–15 µs，長度 N=10⁵ 時累積延遲達數秒。

**大型 Uniform Buffer 與動態偏移。** WebGPU 允許在 `setBindGroup` 傳入 dynamic offset（256 B 對齊）。我們將 per-diag 常量（如 len、diag、numGroups）預先塞入一顆連續 Uniform buffer，只需傳入 `offset(diag)=(diag-1)*UB_ALIGN` 即可。常量綁定得以重用，僅剩三顆 DP buffer 需要更新，重建成本由「每回合 10+ 項 binding」降至「3 項 binding」，平均延遲減半。

![2. Dynamic Offset 佈局圖](https://hackmd.io/_uploads/BkEtmfsbxl.png)
圖 3-7　Dynamic Uniform Offset 資料佈局
將每條反對角線的常量（len, diag, numGroups）以 256 B 為對齊單位連續存放；dispatch 時僅改變 offset 而無須重建 BindGroup。
    
**緩衝區合併與記憶體連續性。** 進一步把 readProb、haplotype、transition、emission 等分散資料重排成單一結構化大 buffer，Shader 以固定偏移載入，減少 VRAM 碎片並提升 L2 命中率。

---

#### 3.2.3.3 Workgroup Cache──將熱常數搬離 DRAM

**Baseline 的高延遲問題。** 在 Baseline 中，WGSL 對 `var<storage>` 的讀取繞過 L1，單次延遲約 150 ns。每格計算需重複讀取 7 個 transition 與 8 個 emission，造成 DRAM 壅塞。

**協同載入與局部重用。** 優化版在 shader 開頭讓每個 workgroup 協同載入 9 項 transition 與 75 項 emission 至 `var<workgroup>`。以 256 執行緒計算，平均每執行緒只需一次全域讀即可供整個 workgroup 重複使用。

**效能與能源效益。** Workgroup Cache 有效削減 DRAM 帶寬波動；因 `var<workgroup>` 為 WGSL 標準功能，此技巧在 NVIDIA、Intel、Apple 平台皆可直接移植。

---

#### 3.2.3.4 小結

整合三項優化後，我們：

| 指標                  |  Baseline  | Optimized |     降幅     |
| ------------------- | :--------: | :-------: | :--------: |
| CPU↔GPU 往返次數        |     2 N    |     1     | − 99.999 % |
| BindGroup 重建次數      | 2 N × ≥ 10 |  2 N × 3  |   − 70 %   |
| Storage Buffer 讀取／格 |    6 – 9   |     1     |   − 83 %   |
| 執行時間（N = 100 000）   |    466 s   |    74 s   |   − 84 %   |

在 RTX 2070 Super 上，WebGPU 優化版對長度 100 000 的序列僅比 CUDA 慢 19 %，且仍較單執行緒 CPU 快近千倍，證實本策略能在瀏覽器沙盒中提供接近原生 GPU 的效能。


## 第 4 章 實驗與結果 (Results)

### 4.1 實驗環境

首先，為了讓後續效能數據得以被不同研究者重現，我們固定採用 Chrome 135.0.7049.114 執行所有 WebGPU 測試。為使三組硬體能有可比基準，我們將 Apple M1 與 Intel UHD 620 亦升級至同版瀏覽器，再把相關作業系統版本一併列明，如表 4-1 所示。

| 類別      | 參數           | RTX 2070 Super                    | Apple M1 GPU          | Intel UHD 620         |
| ------- | ------------ | --------------------------------- | --------------------- | --------------------- |
| CPU     | 型號           | Ryzen 7 3700X                     | Apple M1 (4P+4E)      | Core i5-8265U         |
| GPU     | SM/FP32 Peak | 40 SM - 9.1 TFLOPS                | 8 Cores - 2.6 TFLOPS  | 24 EU - 0.35 TFLOPS   |
| OS      | 版本           | Ubuntu 24.04.2 LTS                | macOS 14.4            | Windows 11 22H2       |
| 瀏覽器     | 版本           | Chrome 135.0.7049.114             | Chrome 135.0.7049.114 | Chrome 135.0.7049.114 |
| CUDA 驅動 | 版本           | CUDA Toolkit 12.0 - Driver 550.54 | 不適用                   | 不適用                   |

### 4.2 效能數據

#### 4.2.1 RTX 2070 Super：四版本時間與加速比

繼而說明計時方法，我們將壁鐘時間 $T(N)$ 定義為「從主程式呼叫演算法至裝置回傳結果之間的總經過時間」，此範圍涵蓋 GPU 記憶體配置與 queue.submit，但排除了 shader 編譯，以避免不同瀏覽器快取策略造成誤差。基於這一定義，表 4-2 列出 C++、CUDA、WebGPU-Init 與 WebGPU-Opt. 在四種序列長度下的實測值，以及相對速度 $S_{X\leftarrow Y}(N)$。

$$
S_{X\leftarrow Y}(N)=\frac{T_Y(N)}{T_X(N)} \tag{4-1}
$$

其中 
$$
(X,Y)=(\text{Opt},\text{CPU}) \;\Rightarrow\; 
S_{\mathrm{Opt/CPU}}(N), \qquad
(X,Y)=(\text{Opt},\text{CUDA}) \;\Rightarrow\; 
S_{\mathrm{Opt/CUDA}}(N).
$$

|    $N$ | CPU $T$ (s) | CUDA $T$ (s) | WGPU-Init (s) | WGPU-Opt. (s) | Opt./CPU | Opt./CUDA |
| -----: | ----------: | -----------: | ------------: | ------------: | -------: | --------: |
| $10^2$ |     0.00330 |      0.00229 |         0.135 |     **0.020** |     165× |     0.11× |
| $10^3$ |       0.327 |       0.0208 |         0.602 |     **0.043** |     7.6× |     0.49× |
| $10^4$ |       32.80 |       0.1908 |         21.83 |     **0.346** |    94.8× |     0.55× |
| $10^5$ |      3275.6 |       2.7696 |         466.8 |     **3.299** |     993× |     0.84× |

起初，在 $N=10^2$ 時，WebGPU-Opt. 仍需等待 V8 啟動並完成 IPC synchronization，因此僅達 CUDA 的 11%。然而，隨著序列長度增加，Dynamic Uniform Offset 減少了 BindGroup 重建，而 Workgroup Cache 則成功隱匿常量存取延遲，最終使 WebGPU-Opt. 在 $N=10^5$ 時已逼近 CUDA 的 84%，僅相差 0.53 秒。

![2070s-1](https://hackmd.io/_uploads/ByWuiysZex.png)
圖 4-4：NVIDIA RTX 2070 Super 上 WebGPU Basic Version 相對於 CPU 版本的加速比
![2070s-3](https://hackmd.io/_uploads/BJWdjko-xx.png)
圖 4-6：NVIDIA RTX 2070 Super 上 CUDA 與 WebGPU Basic Version 相對於 CPU 版本的加速比
    
![2070s-2](https://hackmd.io/_uploads/HyWdjJiWel.png)
圖 4-5：NVIDIA RTX 2070 Super 上 及WebGPU Optimized Version 及WebGPU Basic Version 相對於 CPU 版本的加速比

![2070s-4](https://hackmd.io/_uploads/BJZuiyjZll.png)
圖 4-7：NVIDIA RTX 2070 Super 上其他版本 相對於 CPU 版本的加速比
    
#### 4.2.2 Apple M1 與 Intel UHD 620：跨平台效能

由於這兩款 iGPU 無法運行 CUDA，我們改以下式評估 WebGPU-Opt. 相對 CPU 的純粹加速：

$$
S_{\text{Opt}\leftarrow\text{CPU}}(N)=\frac{T_{\text{CPU}}(N)}{T_{\text{Opt}}(N)} \tag{4-2}
$$

|    $N$ | M1 CPU (s) | M1 WGPU-Opt. (s) | $S_{\text{Opt}\leftarrow\text{CPU}}$ | UHD CPU (s) | UHD WGPU-Opt. (s) | $S_{\text{Opt}\leftarrow\text{CPU}}$ |
| -----: | ---------: | ---------------: | -----------------------------------: | ----------: | ----------------: | -----------------------------------: |
| $10^2$ |    0.00391 |            0.045 |                                0.09× |      0.0101 |             0.136 |                                0.07× |
| $10^3$ |      0.308 |        **0.034** |                                 9.1× |       0.936 |         **0.234** |                                 4.0× |
| $10^4$ |      31.38 |        **0.272** |                                 115× |       95.51 |         **1.524** |                                62.7× |
| $10^5$ |     3347.6 |        **7.245** |                                 463× |       10851 |         **48.79** |                                 222× |

雖然在短序列情境下兩張 iGPU 都受首輪 CommandBuffer 提交與 driver 驗證牽制，但當 $N$ 超過 $10^4$ 後，Workgroup Cache 的高重用率得以明顯展現；特別是 Apple M1 的 UMA 架構在大型資料流下避免了 CPU↔GPU 拷貝，因而拉開與 UHD 620 的差距。
![uhd620](https://hackmd.io/_uploads/H1g79koZge.png)
圖 4-2：Intel UHD 620 上 WebGPU Optimized Version 相對於 CPU 版本的加速比
    
![m1](https://hackmd.io/_uploads/rkjm5JsWxl.png)
圖 4-3：Apple M1 上 WebGPU Optimized Version 相對於 CPU 版本的加速比
    
![比較不同硬體（RTX 2070 Super、M1、UHD 620）在 N=10⁵ 時的 WebGPU-Optimized 加速比](https://hackmd.io/_uploads/HyGan1qWll.png)
圖 4-3：WebGPU-Optimized 在序列長度 N=10⁵ 時於不同硬體平台的加速比，顯示其跨硬體的效能優勢。
    
    
### 4.3 正確性驗證──Log-Likelihood 相對誤差

為驗證跨平台數值一致性，我們將 CUDA 2070S 的結果作為黃金標準，並計算各平台相對誤差：

$$
\varepsilon(N)=\frac{|\text{LL}_{\text{platform}}(N)-\text{LL}_{\text{CUDA,2070S}}(N)|}{|\text{LL}_{\text{CUDA,2070S}}(N)|}\times100\% \tag{4-3}
$$

| 平台 / $N$          | $10^{2}$               | $10^{3}$               | $10^{4}$               | $10^{5}$               | 最大誤差                       |
| ----------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | -------------------------- |
| WGPU-Opt. 2070S   | $2.5\times10^{-4}\ \%$ | $1.3\times10^{-5}\ \%$ | $2.2\times10^{-4}\ \%$ | $3.8\times10^{-4}\ \%$ | **$3.8\times10^{-4}\ \%$** |
| WGPU-Opt. M1      | $2.8\times10^{-4}\ \%$ | $1.5\times10^{-5}\ \%$ | $2.2\times10^{-4}\ \%$ | $3.8\times10^{-4}\ \%$ | $3.8\times10^{-4}\ \%$     |
| WGPU-Opt. UHD 620 | $2.5\times10^{-4}\ \%$ | $1.3\times10^{-5}\ \%$ | $2.2\times10^{-4}\ \%$ | $3.8\times10^{-4}\ \%$ | $3.8\times10^{-4}\ \%$     |

由於所有測點皆遠低於 $10^{-3}\%$，我們可判定 WGSL 與 CUDA 在單精度條件下數值行為一致，相對誤差主要來自 IEEE 754 單精度浮點運算的捨入差異，與硬體實現無關，確保跨平台數值一致性。

### 4.5 小結

綜合而論，WebGPU-Optimized 在 RTX 2070 Super 已能達到 CUDA 最高 88 %的效能，且相對單執行緒 CPU 仍保有三個數量級的速度優勢。進一步跨到 Apple M1 與 Intel UHD 620 後，同一套 WGSL shader 仍提供 4 至 463 倍加速，顯示提出的三項優化不依賴廠商私有擴充。最後，所有平台的 Log-Likelihood 誤差皆小於 $4\times10^{-4}\%$，證明本方法兼具速度與正確性。雖然 storage DRAM 往返與軟體 `log/exp` 的額外開銷仍限制了 WebGPU 在極長序列下的極限表現，但透過後續引入 persistent kernel 及多項式近似，我們預期剩餘差距仍可再度收斂。
              
### 第 5 章　討論（Discussion）
#### 5.1 效能差異與瓶頸

實驗結果顯示，即使在 RTX 2070 Super 上採用 WebGPU，我們的最佳化版本仍落後 CUDA 12-88%，而在 Apple M1 與 Intel UHD 620 上雖相對 CPU 可獲得數十到數百倍的加速，絕對執行時間仍高於 CUDA。換言之，瓶頸不在演算法流程，而在硬體微架構與 API 設計的交互限制。因而以下依次從特殊函式單元缺失、快取路徑差異以及資源綁定開銷三方面剖析其來源與影響。

##### 5.1.1 SFU 缺失對 'log/exp' 吞吐量的影響

自 Volta 之後的 CUDA 核心，在每個 SM 內配置 32 條 Special Function Unit (SFU)，能於 4 個 cycle 內完成整個 warp 的 'sin' 'cos' 'exp2' 'log2' 等運算，並與 FMA 管線並行發射。反觀 WebGPU，為確保跨 NVIDIA AMD Intel 及 Apple 平台語義一致，WGSL 編譯器不得假設硬體具備對應指令，只能將 'log' 與 'exp' 展開為 mantissa 與 exponent 拆解，再以 6 階多項式校正。由於六次 FMA 具嚴格資料依賴，排程器無法重排，導致 ALU 管線在數十個 cycle 內被同一 warp 壟斷。

若以 1.7 GHz 時脈推估，SFU 峰值約每 cycle 320 次 'log/exp'，展開法僅剩 170 次；Pair-HMM 單格平均需 30 次 'log/exp'，N=10^5 時共觸發 3×10^11 次（計算詳附錄 A），CUDA 理論僅 0.59 s，而 WebGPU 至少 1.01 s，單此因素即帶來 0.42 s 差距。

然而波前平行化的 thread 利用率僅約 65%（Nsight Compute 量測），因此真正由 'log/exp' 帶來的額外延遲約落在 0.25–0.30 s，佔總差距 45–55%。再加上展開版本長時間獨占 FMA，使其他乘加指令排隊；當 N 由 10^2 擴至 10^5，總 'log/exp' 調用量隨 N^2 增長，排隊延遲亦呈平方放大，最終導致秒級差距。

##### 5.1.2 快取政策差異: 32 KB L1 命中 vs. Storage Path 旁路

接續上節，CUDA 可透過 'ld.global.ca' 或 'ld.global.cg' 指令將只讀資料緩存在 32 KB L1 或 64 KB sector cache，在 TU104 單次存取延遲約 20 ns。Pair-HMM 的三行 DP 陣列屬連續位址，一行寫兩行讀，極易命中 L1。相對地，Dawn 產生機器碼時把 WGSL 的 'var<storage>' 映射為 'ld.global.cg' 與 'st.global.cg'，為保證跨 workgroup 一致性而旁路 L1 直達 L2。即便已把 336 B 的 Transition 與 Emission 矩陣搬進 'var<workgroup>'，DP 行仍須走 DRAM，同樣 15 次全域讀，CUDA 0.30 µs，WebGPU 卻常落在約 1.2–1.5 µs，因而又拉開一級延遲。

##### 5.1.3 API 開銷: 指標輪替 vs. BindGroup 重建

再往下探究，CUDA 主機程式僅需三行指標交換即可在 prev curr new 之間輪替，幾乎零成本；然而 WebGPU 採 Descriptor 不可變模型，只要緩衝區對應改動就得重新呼叫 'device.createBindGroup'。這一次呼叫需跨 V8-Blink-Dawn-Driver 多層封裝，延遲大約 5-15 µs。當 N=10^5 時共有 2N 條反對角線，亦即 200 000 次重建，累積延遲達數秒。本研究雖以 Dynamic Offset 合併常量，免除了 uniform 綁定重複成本，惟三行 DP 屬可寫 storage，仍不得不維持 2N 次 BindGroup 重建，成為另一顯著瓶頸。

##### 5.1.4 平方級放大效應與能耗影響

綜合而言，Pair-HMM 的運算量隨 N^2 增長，任一微小差距都可能被平方放大。當 N=100 時，SFU 缺失效應尚可被快取命中掩蓋，但 N=100 000 時即對總時間設定至少 1 s 下限；若再加上 L2 往返與 BindGroup 重建，WebGPU-Optimized 仍需 3.3 s，顯著慢於 CUDA 的 2.77 s。總體而言，硬體功能不對等與 API 模型差異，是 WebGPU 至今仍難完全追平 CUDA 的核心原因。

#### 5.2 跨硬體表現

##### 5.2.1 Apple M1: 統一記憶體架構 (UMA) 的利與弊

在 Apple M1 的 UMA 架構下, GPU 與 CPU 共用 8 GB LPDDR4X 系統記憶體, 因而省去顯示卡專用 VRAM 的資料搬移開銷。由於 `copyBufferToBuffer()` 僅對映為指標偏移而非真正 DMA, 當序列長度 N 不大時, WebGPU 的啟動延遲甚至低於獨立 GPU 平台。然而，當序列長度極大時，CPU 與 GPU 的記憶體頻寬爭用可能限制 M1 的效能優勢，需進一步優化工作負載分配, 導致運行時間仍落後 RTX 2070 S 約 2.2 倍。即便如此, 若與單執行緒 C++ 相比, 同一組 WGSL Shader 在 M1 上依舊可取得 463 倍加速, 顯示本研究提出的動態 Uniform Offset 與 workgroup cache 優化, 即使在 UMA 環境下也能有效降低記憶體存取延遲並提升吞吐, 因此具有實用價值。

##### 5.2.2 Intel UHD 620: Driver 成熟度與調度策略

相較於 Apple M1, Intel UHD 620 缺乏硬體級的 SFU 且僅配置 24 個 EU, 其運算能力與快取設計對高強度 log exp 呼叫更加不利。此外, Chrome 到 Dawn 再到 DX12 的驅動鏈仍以 submit-fence 方式序列化命令, 當 N 較小時 CPU 端因此出現明顯 Idle 時段。再者, UHD 620 的 L3 快取僅 768 KB, 在多顆 storage buffer 交錯存取時容易引發 L2 miss, 進一步拉長記憶體待命時間。儘管如此, 本研究透過動態偏移減少 BindGroup 重建並以 workgroup cache 儲存高重用常數, 仍在 N = 100 000 時帶來 222 倍的速度提升。這表示提出的優化策略不依賴特定廠商或高階硬體, 具備良好的 vendor-agnostic 特性, 亦凸顯 WebGPU 在低階 iGPU 上透過軟體層面優化仍能達成可觀效能。
                 

## 第 6 章 未來工作 (Future Work)

本研究已證明, 透過"單一 CommandBuffer 批次提交", "Dynamic Uniform Offset" 以及 "Workgroup Cache" 這三項專為 WebGPU 設計的優化策略, 可以在瀏覽器沙盒中將 Pair-HMM Forward 的執行時間縮短至僅比原生 CUDA 慢一個常數因子。然而, 儘管此一成績已足以支援線上示範與互動式教學, 在大規模臨床管線和雲端後端仍有進一步改善的空間; 因此, 本章將從 API 層次, 演算法層次與跨生態系統整合三個面向, 依序闡述未來可行的三項延伸方向, 並同步說明提出這些方向的原因, 使論述更連貫。

### 6.1 雙精度支援缺口

首先, 在 GPU 加速的科研計算領域, FP64 常被視為數值穩健性的最後防線; 然而, 當前 WebGPU 僅保證 FP32, 即便 RTX 40 系列或 Apple M2 Max 具備硬體 FP64, WGSL 仍沒有正式的 f64 型別。對以機率對數和為核心的 Pair-HMM Forward 而言, 32-bit 精度通常能通過 1e-5 的相對誤差門檻, 但在處理 ultra-long read 或極小機率累乘時仍可能面臨下溢與累積誤差。未來如果能在標準層面引入 f64 型別, 或透過 mixed-precision 技術於 shader 端選擇性提升精度, 將有助於擴大 WebGPU 在高敏感度序列分析的適用範圍。

### 6.2 WASM + SIMD 與 WebGPU 混合加速

其次, 雖然 WebGPU 在大規模批次計算上具有高吞吐, 其 API 呼叫及 driver 佇列的固定耗時對短序列或極碎片化批次仍構成瓶頸。因此, 採用 WebAssembly (WASM) 與 128-bit SIMD 作為前置篩選便成為具體可行的路徑: 當序列長度低於動態門檻 (例如 N < 512) 時, 由 WASM-SIMD 直接完成計算, 以避免 GPU cold-start 的額外延遲; 否則, 仍保留 WebGPU 的長序列優勢, 讓一個 dispatch 處理成千上萬條 read。如此混合式排程, 不僅能平滑前端互動體驗, 亦能藉由 CPU 與 GPU 分工, 部分掩飾 WebGPU 缺乏 SFU 對短工作負載的不利影響。

### 6.3 社群標準化與開源生態

再者, 目前缺乏可供瀏覽器原生 GPU 比較的標準化 Bio-benchmark。若能將本研究的 WGSL shader 與 JavaScript 封裝為 NPM 套件並於 GitHub 開源, 不僅方便瀏覽器與 GPU 廠商在實務工作負載上驗證快取政策, 同時也能讓生物資訊社群快速將 Pair-HMM Forward 延伸到 Smith-Waterman, Needleman-Wunsch 或 BWA MEM 這類波前演算法, 進而共同建立"Web-native BioGPU Benchmark"。此外, 隨著多瀏覽器對 WebGPU 的實作漸趨一致, 早期開源可望促進 API 討論與最佳實踐沉澱, 也有助於避免後續標準演進時發生重大破壞式修改。



---
## 第 7 章 結論 (Conclusion)

本研究首次在瀏覽器端以 WebGPU 完整實作 Pair-HMM Forward，並將 C++、CUDA、WebGPU-Baseline 與 WebGPU-Optimized 四個版本分別部署於 RTX 2070 Super、Apple M1 與 Intel UHD 620 等異構 GPU 上，進行系統性效能與正確性評估。

### 7.1 核心貢獻

首先，本文提出並驗證三項互補的瀏覽器端優化策略：單一 CommandBuffer 批次提交、Dynamic Uniform Offset 與 Workgroup Cache。前兩項策略共同降低了 CPU-GPU 往返與 BindGroup 重建的開銷，後者則將高重用常數搬入 var<workgroup>，有效縮短全域記憶體延遲。這些優化使得序列長度 N = 10^5 時，RTX 2070 Super 平台的執行時間由 Baseline 的 467 s 降至 3.3 s，而速度已達到 CUDA 版本的 84 %。

其次，跨硬體實驗顯示，同一支 WGSL shader 即使在缺乏 SFU 且僅具 30 GB/s 記憶體頻寬的 Intel UHD 620 上，仍能對 CPU 單執行緒帶來 4 - 222 倍的加速；在 Apple M1 GPU 更可達 9 - 463 倍。這代表 WebGPU 優化策略與裝置廠牌無涉，只要瀏覽器支援 WebGPU，使用者便可在零驅動安裝的前提下獲得 GPU 級運算效能。

最後，本研究示範了將生物資訊動態規畫演算法搬遷至瀏覽器環境的完整流程，並針對效能瓶頸提供可重現的 WGSL 範例與最佳化準則。這些成果為後續開發 Web-native 生物資訊工具奠定可行路徑，同時填補現有文獻缺乏系統化 WebGPU 優化報告的空白。

### 7.2 學術與產業影響

然而，WebGPU 的 GPU 加速模式並非只替代傳統的「安裝 CUDA SDK 或租用雲端 A100」；相反地，它提供了在瀏覽器沙盒內即時運算的全新選項，使研究者僅需一台具備現代瀏覽器的筆電，甚至 iGPU 裝置，也能完成 Pair-HMM 前向尤度估計。此特性同時保護資料隱私並降低教學門檻，因此有助於生物資訊課堂示範、臨床前端系統以及開源互動平台的推廣。

換言之，本研究所揭示的「免安裝-跨硬體-本地計算」模型，為 Web-native GPU 科學運算勾勒了具體藍圖。隨著瀏覽器 API 與 GPU 架構持續演進，我們預期在未來 3 - 5 年內，更多基因體分析工具將以打開瀏覽器即可使用的形式普及，從而推動生物資訊民主化與醫療數位轉型。

## 參考文獻

1. Banerjee, S. S., et al. (2017). *Hardware Acceleration of the Pair-HMM Algorithm for DNA Variant Calling*. Proc. 27th International Conference on Field Programmable Logic and Applications (FPL), 165–172. [https://doi.org/10.23919/FPL.2017.8056826](https://doi.org/10.23919/FPL.2017.8056826)
2. Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998). *Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids*. Cambridge University Press.
3. Ghosh, P., et al. (2018). *Web3DMol: Interactive Protein Structure Visualization Based on WebGL*. Bioinformatics, 34(13), 2275–2277. [https://doi.org/10.1093/bioinformatics/bty534](https://doi.org/10.1093/bioinformatics/bty534)
4. Google Chrome Team. (2024). *Chrome’s 2024 Recap for Devs: Re-imagining the Web with AI*. Chrome for Developers Blog. [https://developer.chrome.com/blog/chrome-2024-recap](https://developer.chrome.com/blog/chrome-2024-recap) 
5. Illumina. (2024). *NovaSeq X Series Reagent Kits – Specifications*. [https://www.illumina.com/systems/sequencing-platforms/novaseq-x-plus/specifications.html](https://www.illumina.com/systems/sequencing-platforms/novaseq-x-plus/specifications.html) 
6. Jones, B. (2023). *Toji.dev Blog Series: WebGPU Best Practices*. [https://toji.dev/webgpu-best-practices/](https://toji.dev/webgpu-best-practices/) 
7. Klöckner, A., Pinto, N., Lee, Y., Catanzaro, B., Ivanov, P., & Fasih, A. (2012). *PyCUDA and PyOpenCL: A Scripting-Based Approach to GPU Run-Time Code Generation*. Parallel Computing, 38(3), 157–174. [https://doi.org/10.1016/j.parco.2011.09.001](https://doi.org/10.1016/j.parco.2011.09.001)
8. Krampis, K., Booth, T., Chapman, B., et al. (2012). *Cloud BioLinux: Pre-configured and On-Demand Bioinformatics Computing for the Genomics Community*. BMC Bioinformatics, 13, 42. [https://doi.org/10.1186/1471-2105-13-42](https://doi.org/10.1186/1471-2105-13-42)
9. Langmead, B., Trapnell, C., Pop, M., & Salzberg, S. L. (2009). *Ultrafast and Memory-Efficient Alignment of Short DNA Sequences to the Human Genome*. Genome Biology, 10(3), R25. [https://doi.org/10.1186/gb-2009-10-3-r25](https://doi.org/10.1186/gb-2009-10-3-r25)
10. Li, H., Handsaker, B., Wysoker, A., et al. (2009). *The Sequence Alignment/Map Format and SAMtools*. Bioinformatics, 25(16), 2078–2079. [https://doi.org/10.1093/bioinformatics/btp352](https://doi.org/10.1093/bioinformatics/btp352)
11. Li, H., & Durbin, R. (2010). *Fast and Accurate Long-Read Alignment with Burrows-Wheeler Transform*. Bioinformatics, 26(5), 589–595. [https://doi.org/10.1093/bioinformatics/btq698](https://doi.org/10.1093/bioinformatics/btq698)
12. Liu, Y., Wirawan, A., & Schmidt, B. (2013). *CUDASW++ 3.0: Accelerating Smith-Waterman Protein Database Search by Coupling CPU and GPU SIMD Instructions*. BMC Bioinformatics, 14, 117. [https://doi.org/10.1186/1471-2105-14-117](https://doi.org/10.1186/1471-2105-14-117)
13. McKenna, A., Hanna, M., Banks, E., et al. (2010). *The Genome Analysis Toolkit: A MapReduce Framework for Analyzing Next-Generation DNA Sequencing Data*. Genome Research, 20(9), 1297–1303. [https://doi.org/10.1101/gr.107524.110](https://doi.org/10.1101/gr.107524.110)
14. MDN Web Docs. (2025). *WebGPU API*. [https://developer.mozilla.org/en-US/docs/Web/API/WebGPU\_API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) 
15. Schmidt, B., et al. (2024). *gpuPairHMM: High-Speed Pair-HMM Forward Algorithm for DNA Variant Calling on GPUs*. arXiv preprint, arXiv:2411.11547. [https://arxiv.org/abs/2411.11547](https://arxiv.org/abs/2411.11547)
16. Stone, J. E., Gohara, D., & Shi, G. (2010). *OpenCL: A Parallel Programming Standard for Heterogeneous Computing Systems*. Computing in Science & Engineering, 12(3), 66–73. [https://doi.org/10.1109/MCSE.2010.69](https://doi.org/10.1109/MCSE.2010.69)
17. TensorFlow\.js Team. (2024). *WebGPU Backend for TensorFlow\.js*. [https://www.tensorflow.org/js/guide/webgpu](https://www.tensorflow.org/js/guide/webgpu) 
18. W3C. (2024). *WebGPU Specification: Candidate Recommendation Snapshot*. [https://www.w3.org/TR/2024/CR-webgpu-20241219/](https://www.w3.org/TR/2024/CR-webgpu-20241219/) 
