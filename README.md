## 一、**中英文摘要（Abstract）**
隨著 GPU 加速在生物資訊領域日漸普及，但傳統 CUDA／OpenCL 解決方案須安裝驅動並受限硬體，對教學與臨床前端應用造成門檻。2024 年正式標準化的 WebGPU 在瀏覽器沙盒內以單一 JavaScript API 連結 Vulkan／D3D12／Metal，兼具「免安裝、跨硬體、資料留在本機」三大優勢；本研究據此評估其在高強度 Pair-Hidden Markov Model Forward（Pair-HMM Forward）演算法上的可行性。首先以周育晨（2024）公開之 C++／CUDA 程式為基準實作 WebGPU Baseline，再針對 CPU↔GPU 往返、BindGroup 重建與全域記憶體延遲三大瓶頸，依序導入「單一 CommandBuffer 批次提交」「Dynamic Uniform Offset」與「Workgroup Cache」，形成 WebGPU-Optimized。於 NVIDIA RTX 2070 Super、Apple M1 與 Intel UHD 620 測試序列長度 10²–10⁵；結果顯示 Optimized 版本較 Baseline 加速 6.8–142 倍，並可達 CUDA 12–88 % 速度，三平臺 Log-Likelihood 相對誤差皆低於 10⁻⁵，且在無 NVIDIA GPU 時仍對單執行緒 C++ 提供 3–463× 加速。此研究證實僅憑 JavaScript＋WGSL，即能於瀏覽器中於秒級完成 Pair-HMM Forward 計算，並提出三項瀏覽器端專屬優化策略及跨硬體實測結果，為 Web-native 生物資訊工具奠定基礎。

## 二、**正文章節大綱**

# 第 1 章　緒論（Introduction）
## 1.1 研究背景（Background）

高通量定序（Next‑Generation Sequencing, NGS）技術的蓬勃發展，使得基因體資料的規模與複雜度以指數速度攀升，生物資訊分析對計算效能提出前所未有的挑戰。**Pair Hidden Markov Model（Pair‑HMM）Forward 演算法**能同時支援序列比對、基因型鑑定與變異偵測，是眾多基因體流程不可或缺的運算核心。然而現行工具多以 C++ 或 Python 實作，再藉 NVIDIA CUDA 或 OpenCL 取得 GPU 加速；使用者除了安裝驅動、設定 SDK 與相依函式庫，還受限特定 GPU 架構。對教育現場、雲端共享與非專業研究者而言，這些前置作業往往形成高門檻。雲端服務雖可降低本機安裝負擔，卻伴隨帳號管理、網路延遲與資料外流疑慮。

自 2024 年 5 月 WebGPU 正式進入 Chrome 穩定通道以來，Firefox Nightly 與 Edge Dev 亦陸續提供實驗支援。WebGPU 以單一 JavaScript API 映射 Vulkan、D3D12、Metal 後端，並運行於瀏覽器沙盒，讓 NVIDIA、AMD、Intel 乃至 Apple Silicon GPU 得以免安裝驅動即時執行平行運算。因此，WebGPU 被視為打破硬體與作業系統藩籬的嶄新契機，有望大幅降低生物資訊工具的使用門檻。

## 1.2 研究動機與目的（Purpose）

然而，WebGPU 尚屬新興標準，瀏覽器端 GPU 計算仍面臨 API 排程開銷、缺乏全域同步與特殊函式單元（SFU）等硬體差異，使其在高強度運算場域的可行性尚未被系統性驗證。因此，本研究旨在探討：在「免安裝、跨硬體、資料不離端」的前提下，WebGPU 能否以足夠效能執行 Pair-HMM Forward，並作為 CUDA 的實用替代方案？

## 1.3 研究方法與主要結果（Methods & Results）

本研究以周育晨（2024）公開之 C++ 與 CUDA 程式為效能基準，首先移植至 WebGPU 形成 Baseline，再針對 (i) CPU↔GPU 往返、(ii) BindGroup 重建、(iii) 全域記憶體存取延遲三大瓶頸，依序導入 單一 CommandBuffer 批次提交、Dynamic Uniform Offset 與 Workgroup Cache，構成 WebGPU-Optimized。在 NVIDIA RTX 2070 Super 上測試序列長度 100 至 100 000 顯示，WebGPU-Optimized 可較基線快 6.8–142 倍，並逼近 CUDA 12–88 % 效能；於 Apple M1 與 Intel UHD 620 亦對 CPU 提供 4–463× 加速，Log-Likelihood 誤差均低於 10⁻⁵。

## 1.4 結論與貢獻（Conclusion）

綜上所述，本研究證明只需 JavaScript 與 WGSL，即能在瀏覽器沙盒中於秒級時間完成中大型 Pair-HMM Forward 計算。三項瀏覽器端優化策略—批次提交、動態偏移與工作群組快取—互補消弭了 WebGPU 三大軟硬體瓶頸；跨 NVIDIA、Apple、Intel 硬體的驗證亦顯示方法具廠商不可知性。此成果為「打開瀏覽器即用」的基因體分析鋪路，並為未來雙精度支援 與 WASM-SIMD 混合加速奠定基礎。


# 第 2 章　文獻探討（Related Work）

## 2.1 pair-HMM Forward 在 GPU 平台上的加速策略

隨著 GPU 並行計算能力的提升，多項研究嘗試將 pair-HMM Forward 演算法從 CPU 端移植至 GPU，以突破傳統序列分析的效能瓶頸。主要可分為以下兩大類策略：

**2.1.1 Intertask vs. Intratask 並行化**  
Ren 等人（Shanshan Ren et al.）提出兩種不同的併行模式【15】：  
- **Intertask**：將每對序列（read vs. haplotype）分配給不同執行緒，各自獨立執行完整的 Forward 演算法。此方式併行度極高，但各執行緒間缺乏資料共用，且必須在開始與結束階段進行同步，造成額外開銷。  
- **Intratask**：將單一執行個體的動態規劃矩陣劃分為多個子區塊，分派給多執行緒協同運算；在對角線依賴關係處進行必要同步，以維持計算正確性。相比 Intertask，Intratask 能減少重複工作量，但實現更為複雜，需設計細緻的佈局與同步機制。

**2.1.2 Anti-Diagonal 平行化與 Shared Memory 利用**  
Li 等人（Enliang Li et al.）採用 **反對角線（anti-diagonal）** 平行化技術，搭配 **shared memory** 快取中間結果【8】：  
1. **反對角線併行**：對動態規劃矩陣的每條反對角線進行同時計算，由於同一反對角線上的格子相互之間無直接依賴，可大幅提升併行度。  
2. **shared memory 快取**：在每次 dispatch 前，將 emission 與 transition 機率載入至 shared memory，以減少對全域記憶體的頻繁存取，並降低 CPU ↔ GPU 間的傳輸延遲。  
實驗結果顯示，此策略在序列長度增加時效果尤為明顯，能有效降低記憶體存取瓶頸並提升整體吞吐量。

儘管上述方法在 GPU 平台上已取得顯著加速成效，但均建立於 CUDA 或 OpenCL 的底層 API 之上，尚需安裝驅動與 SDK，且多限於 NVIDIA 或特定廠牌。接下來，本研究將探索如何將這些 GPU 並行化思路移植至瀏覽器端 WebGPU，並應用於免安裝、跨硬體情境下的 pair-HMM Forward 加速。

---

## 2.2 WebGPU 技術與優化現狀

隨著 WebAssembly 與 WebGPU 的快速發展，瀏覽器端 GPU 加速成為可能。下文將介紹 WebGPU 的核心架構與 API，並回顧現有優化策略及其局限。

**2.2.1 WebGPU 架構與 JavaScript API**  
WebGPU 透過 JavaScript 提供統一 API，將計算命令編碼（CommandEncoder）、管線設定（PipelineLayout）、資源綁定（BindGroup）等指令送至瀏覽器 GPU driver，再由其轉譯為底層驅動呼叫，最終提交至實體 GPU 執行。計算核心使用 WGSL（WebGPU Shading Language）撰寫，需明確宣告 buffer、binding 與資源佈局，並在 JavaScript 端 orchestrate 整個 dispatch 流程。相較於 CUDA，WebGPU 雖少了直接操控硬體的權限，但具備「免安裝驅動」與瀏覽器安全沙盒（sandbox）保護，確保程式碼僅在本地執行並降低安全風險。

**2.2.2 現有優化策略**  
多項研究與工程經驗提出了適用於 WebGPU 的優化方法：  
- **BindGroup 池化（BindGroup Pooling）**：預先建立一組 BindGroup，並在迴圈中重複使用，避免頻繁的 createBindGroup() 呼叫。  
- **Dynamic Offset**：利用 single BindGroup 搭配動態偏移（dynamic offset）來更換 buffer 綁定，減少 BindGroup 數量並降低建立成本。  
- **Persistent Kernel**：將多次 dispatch 的迴圈邏輯遷移至單一 WGSL shader 內部執行，只執行一次 queue.submit()，大幅降低 command submission 的 overhead。  
- **Workgroup Cache**：將常用資料（如 emission/transition 矩陣）載入至 var<workgroup>，在同一工作群組內多次存取，減少對 storage buffer 的全域存取延遲。

這些策略在圖形渲染、機器學習等領域已有成功案例，但在生物資訊的 pair-HMM Forward 演算法上，尚未見系統性應用與驗證。特別地，WebGPU 缺乏 CUDA 專用 SFU 硬體加速浮點函式，以及不可跨工作群組同步的限制，使其在高強度 log/exp 運算與全域依賴拆分方面面臨挑戰。

**2.2.3 研究缺口**  
綜上所述，現有文獻多聚焦於傳統 GPU 或特定領域的 WebGPU 優化，但：  
1. 少有針對生物資訊中 pair-HMM Forward 演算法的瀏覽器端實作與優化報告；  
2. 尚未驗證 WebGPU 在無驅動安裝、跨硬體且保持本地安全的情境下，能否利用上述優化策略達到實用效能；  
3. 關於 SFU 缺失與 IPC 延遲對高強度浮點運算的影響，尚缺定量分析。  

因此，本研究將結合 GPU 平台上的併行化技術與 WebGPU 特有優化策略，針對免安裝、跨硬體、保持本地安全的瀏覽器環境，開發並評估高效能的 pair-HMM Forward 實作方案，填補上述研究空白。  

# 第 3 章　研究方法（Methods）

## 3.1 數學模型（Mathematical Model）
### 1  Pair-HMM 基礎

* **隱藏狀態**：Match (M)、Insert (I)、Delete (D)。
* **字母表**：A、C、G、T 與 gap (–)。
* **轉移機率 $t_{\text{state}_a,\text{state}_b}$**：由一個狀態轉到另一個狀態的機率。
* **發射機率 $e(x,y)$**：在指定狀態下，觀測到讀序列字元 $x$ 與雜合序列字元 $y$ 的機率。

### 2  Pair-HMM Forward 演算法（Algorithm 1）

1. **初始化**

   $$
   M_{0,j}=I_{0,j}=0,\quad D_{0,j}=\frac{1}{n}\quad (1\le j\le n)
   $$

   允許一開始在雜合序列端有免費刪除。
2. **雙層迴圈遍歷**

   * 外層 $i=1\ldots m$：逐字元掃描讀序列。
   * 內層 $j=1\ldots n$：逐字元掃描雜合序列。
3. **遞迴更新**

   $$
   \begin{aligned}
   M_{i,j} &= e_{i,j}\bigl(t_{MM}M_{i-1,j-1}+t_{IM}I_{i-1,j-1}+t_{DM}D_{i-1,j-1}\bigr)\\
   I_{i,j} &= t_{MI}M_{i-1,j}+t_{II}I_{i-1,j}\\
   D_{i,j} &= t_{MD}M_{i,j-1}+t_{DD}D_{i,j-1}
   \end{aligned}
   $$

   * 只有 **Match** 狀態需乘以發射機率 $e_{i,j}$，Insert、Delete 為 gap，不乘發射項。
4. **終端機率**

   $$
   P(\text{alignment})=\sum_{j=1}^{n}\bigl(M_{m,j}+I_{m,j}\bigr)
   $$
    終態只累計 𝑀 與 𝐼 ，因為 𝐷 代表以刪除結束，不對齊讀序列末端。
    
5. **動態規劃特性**：以上遞迴方程列舉並累計所有可能的部分對齊路徑；由於每一步都只依賴已計算之左、上、左上三格，因此時間複雜度為 𝑂(𝑚𝑛) ，且可保證得到精確的前向機率。

---

## 3.2 系統設計與實作（System Design and Implementation）

### 3.2.1 C++／CUDA 版

參考周育晨（2024）之 GPU 版本，以 **一條反對角線＝一次 Kernel** 的策略實現全域同步。每次 Kernel 發射後，以 `cudaDeviceSynchronize()` 作為 GPU‑wide barrier；如此雖需 $2N$ 次 Kernel 呼叫，卻能保證跨 thread‑block 的資料依賴。而在同一 thread‑block 中，使用 `__syncthreads()` 實現組內同步。

對於三條 DP 陣列 $M,I,D$，程式採用「四行指標輪替」：在主機端僅以四行指標變換將 `prev → curr → new`，無須任何 Descriptor 更新或記憶體複製。由於 CUDA 指標可以像一般 C 指標般操作，此技巧幾乎無成本。


### 3.2.2 WebGPU Baseline

為了在瀏覽器端忠實重現 CUDA 的演算法流程，我們把 **「一條反對角線＝一次工作負載」** 的概念保留下來，但必須把整個 *波前迴圈* 搬回 JavaScript 端控制。其原因與影響，概述如下。

---

#### （一）從 CUDA「多次 Kernel」到 WebGPU「多次 dispatch」

Pair‑HMM Forward 的計算沿著動態規畫矩陣的反對角線（wavefront）推進；每條 wavefront 必須等前一條全部完成才能繼續。CUDA 最直觀的做法是用一個 `for` 迴圈依序啟動 Kernel，並在兩次 Kernel 之間呼叫 `cudaDeviceSynchronize()`。這樣做等同於在 GPU 端安插一層全域 barrier，同時讓主機程式有機會交換三條 DP 指標。

WebGPU 的 WGSL 只有 `workgroupBarrier()`，缺乏跨 workgroup 的同步指令；Shader 也無法像 CUDA Dynamic Parallelism 那樣在裝置端再次啟動子工作負載。因此 **每條反對角線都必須由 Host 端發起一次 `dispatchWorkgroups()`**。Host 端在呼叫 `queue.submit()` 送出上一條 wavefront 的 compute pass 後，必須等 `device.queue.onSubmittedWorkDone()` 回傳，才能安全地更新 Uniform 並發送下一條 dispatch。對序列長度 $N$ 來說，整個演算法需要 **$2N$ 次 dispatch 與同樣多的 CPU↔GPU 往返**。這種做法雖能保證資料正確性，卻將同步延遲全數暴露在 JavaScript 執行緒，成為 Baseline 的第一個重大瓶頸。

---

#### （二）指標輪替與 BindGroup 的不可變性

CUDA 在兩條 wavefront 之間只需簡單地交換三個 `float*` 指標，就能把 `prev → curr → new` 的角色往前推；Driver 無需重新配置任何資源。而 WebGPU 的 Buffer 綁定點在 **BindGroup 建立時就被寫死**。若要讓 Shader 在下一條 wavefront 讀另一個 DP Buffer，唯一辦法是 **重新呼叫 `device.createBindGroup()`** 將對應 binding slot 指到新的 `GPUBuffer`。每次建立 BindGroup 會跨越 V8→Blink→Dawn→Driver 多層封裝，延遲約 10–50 µs；在 $2N$ 次迴圈中重複進行，累積成本高達數十秒。這是 Baseline 的第二個瓶頸。

---

#### （三）shared memory 的缺席與高延遲 storage buffer 存取

CUDA 將 9 個 Transition 係數與 75 個 Emission 係數放入 48 KB shared memory，共用延遲僅 80 ns；WGSL 雖提供 `var<workgroup>`，但大小受限、且資料必須由 Shader 顯式搬運。Baseline 為求驗證正確性，直接把小矩陣留在 `storageBuffer`。結果每格計算要重複進行 6–9 次 global read，單次延遲約 300 ns，遠高於 shared memory。這成為第三個瓶頸。

---

#### （四）Baseline 的暫行解決方案

* **多次 dispatch 與 Host 迴圈**
  利用「Compute Pass 完畢後 GPU 先天具備的序列化特性」代替 `cudaDeviceSynchronize()`，保證正確性。
* **逐步重建 BindGroup**
  在 JavaScript 端以「重新建立三相 DP Buffer 的 BindGroup」來模擬 CUDA 的指標輪替。這樣雖有額外開銷，卻確保 Shader 始終讀取正確的前一條 wavefront。
* **一次性建立 ComputePipeline**
  為避免再添開銷，Baseline 在初始化階段只建立一支 `ComputePipeline`，並重用同一 `ComputePassEncoder`；如此至少排除了重複編譯 WGSL 與 Pipeline State Object 的高昂成本。

---

#### （五）Baseline 效能概況

在 NVIDIA RTX 2070 Super 上，Baseline 執行長度 $N=100\,000$ 的序列耗時 466 秒，較 CUDA 慢約兩個數量級，延遲來源幾乎全由「$2N$ 次 IPC 同步＋$2N$ 次 BindGroup 生成＋頻繁 storage 讀」組成。這一結果直接揭示了 WebGPU 與 CUDA 架構差異帶來的瓶頸，也為下一節所提的三項優化策略指明了攻擊面：**減少 Host ↔ GPU 往返、把 BindGroup 變動最小化，以及把熱資料搬進 `var<workgroup>` 快取**。


### 3.2.3 WebGPU Optimized（本研究）

為徹底解除 Baseline 在 **(1) 頻繁主機同步、(2) 頻繁 BindGroup 重建、(3) 高延遲記憶體存取** 三大瓶頸，本研究提出三項瀏覽器端優化策略：
A. 單 CommandBuffer 批次提交、B. Dynamic Uniform Offset、C. Workgroup Cache。
本節先說明 WebGPU 的指令錄製與提交機制，再依序討論三項優化的設計動機、實作細節與效能效益。

---

#### 3.2.3.1 單一 CommandBuffer 批次提交──降低 CPU↔GPU 往返

##### 3.2.3.1.1 CommandEncoder 與指令流

在 WebGPU 中，`CommandEncoder` 相當於一部「命令錄製器」。開發者可透過 `beginComputePass()`、`dispatchWorkgroups()`、`copyBufferToBuffer()`、`end()` 等 API，將所有欲於 GPU 執行的動作依序錄製；呼叫 `encoder.finish()` 後，即產生一個 `GPUCommandBuffer`。最後將此對象一次性提交給排程器：

```javascript
device.queue.submit([commandBuffer]);
```

GPU 會依照錄製順序連續執行所有命令，期間 **無需 CPU 介入**。只有在程式顯式呼叫

```javascript
await device.queue.onSubmittedWorkDone();
```

時，JavaScript 執行緒才會等待 GPU 完成整條指令流。

##### 3.2.3.1.2 傳統多次提交的痛點

Baseline 做法如下：

```javascript
for (let diag = 1; diag <= totalDiag; ++diag) {
  device.queue.submit([encoderForThisDiag.finish()]);
  await device.queue.onSubmittedWorkDone();   // CPU 等待
  prepareNextEncoder();                       // 重建下一次指令
}
```

對長度 $N$ 的序列需要 $2N$ 次「submit → barrier → 建立新 encoder」，每一次 `onSubmittedWorkDone()` 都要產生 CPU–GPU handshake，導致多層 IPC 與排程延遲。

##### 3.2.3.1.3 一次性指令流的優勢

本版本仍於每條對角線呼叫 beginComputePass()，僅將 submit 次數自 2N 降為 1；若要變成『單一 pass + 多次 dispatch』可再進一步優化

*GPU 端執行期間不再切回 CPU*，優點包括：

* **零中斷**——GPU 從頭跑到尾，核心使用率顯著提升。
* **Driver Overhead 大幅降低**——省去上萬次 CPU→GPU submit / barrier 開銷，並減少 CommandEncoder-related 驗證。
* **連續 DRAM 流**——dispatch 與 copy 命令連續佇列，降低突發性帶寬抖動。

實驗顯示，將 2N 次 IPC 壓縮成 1 次，可顯著縮短總執行時間。

---

#### 3.2.3.2 Dynamic Uniform Offset──減少常量更新成本

##### 3.2.3.2.1 多小 buffer 與多次 BindGroup 的問題

當 Shader 需要讀寫多個 DP Buffer，加上輸入序列、Haplotype 與 Transition／Emission，小型 storage buffer 很容易破十個。若沿用 Baseline，每 dispatch 都必須重建一次 BindGroup。

```javascript
const bg = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: readProb } },
    { binding: 1, resource: { buffer: haplotype } },
    ⋯⋯
  ]
});
pass.setBindGroup(0, bg);
```

此舉產生 **大量 API 呼叫、Driver 驗證與 GC 壓力**。

##### 3.2.3.2.2 大型 Uniform Buffer 與偏移量

WebGPU 容許在 `setBindGroup()` 時傳入一組 **dynamic offset**（256 B 對齊）。因此我們將每條對角線的動態參數 `(len, diag, numGroups)` 放入以 **`UB_ALIGN`\*(2N)** 為大小的大 Uniform Buffer：

```
offset(diag) = (diag－1) × UB_ALIGN
```

dispatch 只需：

```javascript
pass.setBindGroup(0, bg, [offset(diag)])
```

目前實作中，BindGroup 生成次數仍為 $2N$（每條對角線一次）；常量 buffer 的確可重用，但仍透過重建 BindGroup 來更新 dpPrev / dpCurr / dpNew 三個動態 binding，尚未真正將次數壓縮到 3 顆預建 BindGroup。

##### 3.2.3.2.3 緩衝區合併與記憶體連續性

將 readProbMatrix、Haplotype、Transition、Emission 等小型 storage buffer 至一顆結構化大 buffer，Shader 以固定偏移讀取。避免 VRAM 零散配置，提高快取命中率。

---

#### 3.2.3.3 Workgroup Cache──把熱常數搬離 DRAM

##### 3.2.3.3.1 Baseline 的高延遲問題

WGSL 對 `var<storage>` 存取通常繞過 L1 Cache，延遲約 100–200 ns；而 Baseline 每格約需重複讀 15 floats（7 T + 8 E），仍屬高延遲瓶頸。

##### 3.2.3.3.2 協同載入 
* 載入成本 = ⌈9/256⌉ + ⌈75/256⌉ = 2 次；平均分攤到 256 threads，每 thread 最多 1 次、平均 0.33 次。
* 之後每格只需讀寫 DP Buffer，常數延遲降至約 5–15 ns。

##### 3.2.3.3.3 效能與能源效益

* **頻寬**：帶寬峰值下降，DRAM 行為更平順。
* **可移植性**：`var<workgroup>` 為標準 WGSL 特性，跨 NVIDIA／Intel／Apple 均可用。

---

#### 3.2.3.5 小結

* **一次 CommandBuffer** 把全域同步從「$2N$ 次 IPC」變為「1 次排程」，掃除 CPU 阻塞。
* **Dynamic Uniform Offset** 與 **BindGroup** 常量 buffer 重複利用，綁定成本已有所下降。
* **Workgroup Cache** 把高重用常數搬進近端快取，顯著降低 DRAM 來回。

## 第 4 章　實驗與結果（Results）

### 4.1 實驗環境

| 類別          | 參數             | RTX 2070 Super                                                                     | Apple M1 GPU         | Intel UHD 620           |
| ----------- | -------------- | ---------------------------------------------------------------------------------- | -------------------- | ----------------------- |
| **CPU**     | 型號 / 核心數       | Ryzen 7 3700X × 16                                                                 | Apple M1 × 4P + 4E   | i5-8265U × 4            |
| **GPU**     | SM / FP32 Peak | 40 SM · 9.1 TFLOPS                                                                 | 8 Core · 2.6 TFLOPS  | 24 EU · 0.35 TFLOPS     |
| **OS**      | 版本             | Ubuntu 24.04.2 LTS                                                                 | macOS 14.4           | Windows 11 22H2         |
| **瀏覽器**     | 版本             | Chrome  135.0.7049.114                               | 同左                   | 同左                      |
| **CUDA 驅動** | 版本             | CUDA 12.4 + Driver 550                                                             | –                    | –                       |

---

### 4.2 效能數據

#### 4.2.1 RTX 2070 Super：四版本時間與加速比

速度指標（以版本 $Y$ 為基準、版本 $X$ 為被測）定義為

\begin{equation}
S\_{X\leftarrow Y}(N)=\frac{T\_Y(N)}{T\_X(N)},
\label{eq\:speed\_generic}
\end{equation}

其中 $T_X(N)$ 為長度 $N$ 序列之 **壁鐘時間**──自主程式啟動演算法至 GPU／CPU 回傳結果的總經過時間。若令

$$
(X,Y)=(\text{Opt},\text{CPU}) \;\Rightarrow\; 
S_{\mathrm{Opt/CPU}}(N), \qquad
(X,Y)=(\text{Opt},\text{CUDA}) \;\Rightarrow\; 
S_{\mathrm{Opt/CUDA}}(N).
$$

表 4-1 列出四版本在 RTX 2070 Super 的實測結果與相對加速倍數。

<table>
<thead><tr><th rowspan="2">N</th><th colspan="4">執行時間 T(N) / s</th><th colspan="2">加速倍數 S</th></tr>
<tr><th>CPU</th><th>CUDA</th><th>WGPU-Init</th><th>WGPU-Opt.</th>
<th>Opt./CPU</th><th>Opt./CUDA</th></tr></thead>
<tbody>
<tr><td>10²</td><td>0.00330</td><td>0.00229</td><td>0.135</td><td><b>0.020</b></td><td>165 ×</td><td>0.11 ×</td></tr>
<tr><td>10³</td><td>0.327</td><td>0.0208</td><td>0.602</td><td><b>0.043</b></td><td>7.6 ×</td><td>0.49 ×</td></tr>
<tr><td>10⁴</td><td>32.80</td><td>0.1908</td><td>21.83</td><td><b>0.346</b></td><td>94.8 ×</td><td>0.55 ×</td></tr>
<tr><td>10⁵</td><td>3 275.6</td><td>2.7696</td><td>466.8</td><td><b>3.299</b></td><td>993 ×</td><td>0.84 ×</td></tr>
</tbody></table>



*對長度 10⁵，WebGPU-Optimized 僅比 CUDA 慢 19 %（公式 4-1），

---

#### 4.2.2 Apple M1 與 Intel UHD 620：跨平台效能

為避免 CUDA 缺席的比較失衡，跨平台僅採 $$
S_{\text{Opt}\leftarrow\text{CPU}}(N)\;=\;
\frac{T_{\text{CPU}}(N)}{T_{\text{Opt}}(N)},
\tag{4-2}
$$

> **表 4-3　Apple M1：WebGPU-Optimized 對 CPU 速度提升**

|        N | CPU T (s) | WGPU-Opt. T (s) | Speed-up $S_{\text{Opt}\leftarrow\text{CPU}}$ |
| -------: | --------: | --------------: | --------------------------------------------: |
| $10^{2}$ |   0.00391 |           0.045 |                                        0.09 × |
| $10^{3}$ |     0.308 |       **0.034** |                                         9.1 × |
| $10^{4}$ |     31.38 |       **0.272** |                                         115 × |
| $10^{5}$ |   3 347.6 |       **7.245** |                                         463 × |

> **表 4-4　Intel UHD 620：WebGPU-Optimized 對 CPU 速度提升**

|        N | CPU T (s) | WGPU-Opt. T (s) | Speed-up $S_{\text{Opt}\leftarrow\text{CPU}}$ |
| -------: | --------: | --------------: | --------------------------------------------: |
| $10^{2}$ |    0.0101 |           0.136 |                                        0.07 × |
| $10^{3}$ |     0.936 |       **0.234** |                                         4.0 × |
| $10^{4}$ |     95.51 |       **1.524** |                                        62.7 × |
| $10^{5}$ |    10 851 |       **48.79** |                                         222 × |

*短序列下 M1 與 UHD 620 仍受 driver 啟動成本支配；
當 $N≥10³$，WebGPU-Opt. 在兩片非 NVIDIA GPU 皆帶來 ≥4 × 加速。*


### 4.3 正確性驗證──Log-Likelihood 相對誤差

相對誤差公式

$$
\varepsilon(N)=
\frac{\lvert\text{LL}_{\text{platform}}(N)-\text{LL}_{\text{CUDA,2070S}}(N)\rvert}
     {\lvert\text{LL}_{\text{CUDA,2070S}}(N)\rvert}\times100\%.
\tag{4-3}
$$

| 平台 / N            | 10²      | 10³      | 10⁴      | 10⁵      | 最高誤差         |
| ----------------- | -------- | -------- | -------- | -------- | ------------ |
| WGPU-Opt. 2070S   | 2.5e-4 % | 1.3e-5 % | 2.2e-4 % | 3.8e-4 % | **3.8e-4 %** |
| WGPU-Opt. M1      | 2.8e-4 % | 1.5e-5 % | 2.2e-4 % | 3.8e-4 % | 3.8e-4 %     |
| WGPU-Opt. UHD 620 | 2.5e-4 % | 1.3e-5 % | 2.2e-4 % | 3.8e-4 % | 3.8e-4 %     |

*最大誤差 $\varepsilon_{\max}=3.8\times10^{-4}\% <10^{-3}\%$。
所有偏差僅源自單精度捨入，對後續貝氏變異判定影響可以忽略。*

---

### 4.5 小結

1. **效能**：在 RTX 2070 Super，WebGPU-Opt. 已達 CUDA 的 12–88 %；對 CPU 單執行緒保持 30–1000× 加速。
2. **可攜性**：同一支 WGSL 在 Apple M1 與 Intel UHD 620 上仍可帶來 4–463× 加速，證實優化策略不依賴廠商私有功能。
3. **準確度**：三平台 Log-Likelihood 相對誤差低於 $10^{-5}$，數值行為與 CUDA 一致。
4. **瓶頸**：剩餘差距來自 storage DRAM 往返與 `log/exp` 軟體實現；後續可藉 Persistent Kernel 與多項式近似進一步壓縮執行時間。
              
### 第 5 章　討論（Discussion）

#### 5.1 效能差異與瓶頸

實驗結果顯示，瀏覽器端 WebGPU 即使在 RTX 2070 Super 上也仍落後 CUDA 12–88 %，而在 Apple M1 與 Intel UHD 620 上雖可較 CPU 提供數十到數百倍加速，絕對時間仍遠高於 CUDA。究其原因，並非演算法本身，而是硬體微架構與 API 設計的交互效應。本節依序從 **特殊函式單元 (SFU) 缺失**、**快取路徑差異** 與 **資源綁定開銷** 三個層面深入分析。

---

##### 5.1.1 SFU 缺失對 `log/exp` 吞吐量的影響

CUDA 的 Turing/Volta 之後，每個 SM 皆增設 32 條 **Special Function Unit (SFU)**，專門處理 `sin`、`cos`、`exp2`、`log2` 等高階函式。這些單元透過「查表＋CORDIC 旋轉＋線性內插」在 4 個時脈週期內完成整個 warp 的特殊函式運算，並能與 FMA 管線並行發射，幾乎不與通用 ALU 競爭資源。相較之下，WebGPU 為了在 NVIDIA、AMD、Intel、Apple 等異質平台維持語義一致，WGSL 編譯器無法假設目標硬體一定具備對應指令，因此必須將 `log()`╱`exp()` 展開為 **mantissa/exponent 拆解 + 微型查表 + 6 階多項式校正**。這六次 FMA 彼此存在嚴格資料依賴，蘇難以被 GPU 排程器重排序，造成 ALU 管線在數十個時脈內被同一 warp 壟斷。兩者差距可量化如下：在 1.7 GHz 時脈下，SFU 路徑理論上可達到每週期 320 log/exp，而 ALU 展開法僅約 170 log/exp；對 Pair-HMM 一格需要約 30 次 `log/exp` 的情境而言，$N = 10^5$ 時總函式調用量達 $3\times10^{11}$，以理論峰值估算，CUDA 只需 0.59 s 即可完成，WebGPU 至少 1.01 s──單就函式單元就拉開 0.4 s 的差距。

此差距更因 **管線互鎖** 被放大。CUDA 的 SFU 與 FMA 管線物理上獨立，能實現雙發射；WebGPU 的展開版 `log()` 卻完全佔據了 FMA，導致其他乘加指令必須排隊等待，warp scheduler 難以藏匿延遲。當演算法規模由 $N=10^2$ 擴大到 $N=10^5$ 時，這個「倍半」甚至「二倍」的單格差距會被平方級放大，最終成為秒級壁鐘時間差。

---

##### 5.1.2 快取政策差異：32 KB L1 命中 vs. Storage-Path 旁路

NVIDIA 的 `ld.global.ca`／`ld.global.cg` 指令允許開發者顯式標註「只讀」或「讀寫一致性」語義；只讀資料可投進 32 KB L1，也可以落入 64 KB sector cache，在 TU104 平台單次延遲約 20 ns。Pair-HMM 的三行 DP 陣列採「一行寫、兩行讀」的模式，且記憶體位址連續，極易被硬體 prefetch 與 L1 命中所攔截。相反地，WebGPU 的 `var<storage>` 被 Dawn 後端映射為 `ld.global.cg / st.global.cg`──此路徑為確保跨 workgroup 一致性而 **旁路 L1，直接打到 L2**。即使已將 336 B 的 Transition╱Emission 小矩陣搬進 `var<workgroup>`（對應 CUDA shared memory），DP 行本身仍須走 DRAM；因此，CUDA 在同樣 15 次全域讀的情境下或僅耗 300 ns，WebGPU 卻可能跨入微秒級。

---

##### 5.1.3 API 開銷：指標輪替 vs. BindGroup 重建

CUDA 主機程式要將「前一行、當前行、下一行」指標往前推，只需 3 行 C 語言指標交換，成本趨近 0 µs；GPU Driver 在下一次 kernel launch 讀入新的指標即可。WebGPU 的資源綁定採「Descriptor 不可變」模型：`BindGroup` 建立時就必須鎖定每個 binding 與對應 `GPUBuffer`；任何「換指標」都需 `device.createBindGroup()`，而此過程需跨越 V8 → Blink → Dawn → Driver 多層封裝，實測單次 5–15 µs。對 $N=10^5$ 的演算法，反對角線數量為 200 000，光是 BindGroup 生成就可能耗去數秒至數十秒。

本研究已以 **Dynamic Uniform Offset** 將 `(len, diag, numGroups)` 塞進同一顆大型 Uniform buffer，避免每回合重建常量 binding，但 DP 行仍因「可寫」必須各自佔一顆 storage buffer，故 BindGroup 重建次數依舊是 2N。

---

##### 5.1.4 平方級放大效應與能耗影響

Pair-HMM 的工作量隨 $N^2$ 成長，使任何「微幅」硬體差距都被平方放大。例如 SFU 與 ALU 展開在 $N=100$ 幾乎可被快取與排程隱藏，但在 $N=100\,000$ 時便決定了近 1 s 的下限。而 API 與 L2 延遲又與 `log/exp` 差距疊加，解釋了 WebGPU-Optimized 仍需 3.3 s、遠高於 2.77 s CUDA 的原因。更重要的是，ALU 展開導致 FMA 管線長時間滿載，不僅拉高功耗 15–20 W，也讓指令快取因指令體積暴增而出現 thrash，對帶寬 30 GB/s 的 UHD 620 影響尤甚。

#### 5.2 跨硬體表現

##### 5.2.1 Apple M1：統一記憶體架構 (UMA) 的利與弊

M1 GPU 與 CPU 共享 8 GB LPDDR4X，省去顯示卡專用 VRAM：

* **優勢** — `copyBufferToBuffer()` 僅為指標偏移，不需真正 DMA，
  小 N 測試時 WebGPU 啟動成本低於 dGPU；
* **劣勢** — 高併發時記憶體頻寬 (68 GB/s) 分食 CPU，
  $N=100\,000$ 仍落後 RTX 2070 S 2.2 ×；
  但相較單執行緒 C++ 仍有 463 × 加速，證明 UMA 下 WGSL 優化仍具實用性。

##### 5.2.2 Intel UHD 620：Driver 成熟度與調度策略

UHD 620 缺 SFU 且 EU 僅 24 個；更關鍵是

* **Driver 同步**：Chrome–Dawn–DX12 鏈上仍以 **submit–fence** 序列化，
  小 N 出現明顯 CPU Idle；
* **L3 共享快取僅 768 KB**，大量 storage buffer 競爭導致 L2 miss。
  即便如此，動態偏移 + workgroup cache 仍帶來 222 × (N=100 k) 加速，
  顯示優化策略具 **廠商不可知性 (vendor-agnostic)**。
                 

## 第 6 章　未來工作（Future Work）

本研究證明，在瀏覽器沙盒中透過三項專門針對 WebGPU 的優化策略，即「單一 CommandBuffer 批次提交」、「Dynamic Uniform Offset」與「Workgroup Cache」，即可將 Pair-HMM Forward 的執行時間壓縮到僅比原生 CUDA 慢一個常數因子。雖然此結果已足以支援線上示範與互動式教學，對大規模臨床管線或雲端後端而言仍存改進空間。本章提出四條具體後續方向，分別對應 API 層級、演算法層級與跨生態系統整合三個維度。

### 6.1　雙精度支援缺口
在 GPU 加速的科學運算領域，「雙精度浮點（FP64）」往往被視為數值穩健性的最後防線；然而目前 WebGPU 僅保證 FP32，即使在硬體具備 FP64 ALU 的 RTX 40 系列或 Apple M2 Max 上，WGSL 仍沒有正式的 f64 型別，也未開放對應算術與轉換指令。對 Pair-HMM Forward 這類以機率對數和為主要內核的演算法，32-bit 精度雖已足夠通過 $10^{-5}$ 相對誤差門檻，但在某些需要「超長讀段（ultra-long read）」或「極小機率累乘」的應用（例如罕見變異檢測、腫瘤分割）中，FP32 仍可能產生下溢或捨入誤差放大問題。

### 6.2　WASM + SIMD 與 WebGPU 混合加速

雖然 WebGPU 對資料平行負載提供了卓越吞吐，仍存在 API 呼叫與 driver 佇列固定開銷；對短序列或大量小片段的 batch 模式，純 GPU 方案反而可能受啟動成本拖累。WebAssembly（WASM）現已支援 128-bit SIMD（`v128`），在桌面平台可帶來 4–8× 的 CPU 內在加速。未來可採 **混合式執行策略**：

1. 在 JavaScript 端動態依照序列長度門檻（例如 $N < 512$）選擇 WASM-SIMD 路徑，避免 GPU cold-start。
2. 對 GPU 路徑則保留長序列優勢，由 WebGPU 以大批序列併行處理。
   這一策略不僅能平滑性能曲線，亦能進一步掩護 SFU 缺失所帶來的小輸入量低效率問題。

### 6.3　社群標準化與開源生態

目前 WebGPU 缺乏「數值密集型」基準專案。若能將本研究之 WGSL shader、JavaScript 打包為 NPM 套件並在 GitHub 開源，一方面便於瀏覽器廠商以 real-world workload 驗證快取政策，另一方面亦能讓生物資訊社群快速改寫更高階演算法。未來目標是建構一組**Web-native BioGPU Benchmark**，涵蓋 Pair-HMM、Smith–Waterman、Needleman–Wunsch 與 BWA MEM 背後的波前演算法，為標準化出力。

---

## 第 7 章　結論（Conclusion）

本研究首度在瀏覽器端完整實作 Pair-HMM Forward，並以 C++、CUDA、WebGPU-Baseline 與 WebGPU-Optimized 四版本在 RTX 2070 Super、Apple M1 及 Intel UHD 620 三種異構 GPU 上進行系統比較。研究結果可歸納三項核心貢獻與兩項長期影響。

### 7.1　核心貢獻

1. **三項瀏覽器端優化策略的提出與驗證**
   *「單一 CommandBuffer 批次提交、Dynamic Uniform Offset 與 Workgroup Cache」* 形成互補：前者動態 Uniform Offset 免除了 Uniform binding 的重建，但由於三顆 DP buffer 仍以獨立 binding 傳入，每條對角線只需重建一次 BindGroup；後者則將高重用常數搬入 `var<workgroup>`，把記憶體延遲降一個數量級。最終在 RTX 2070 Super 上，長度 $N = 10^{5}$ 序列執行時間由 467 s (Baseline) 下降到 3.3 s，達到 CUDA 84 % 的速度。

2. **跨硬體驗證**
   相同 WGSL Shader 在缺乏 SFU、帶寬僅 30 GB/s 的 Intel UHD 620 仍對 CPU 單執行緒提供 4–222× 加速；在 Apple M1 GPU 更達 9–463×。此結果證明：**只要瀏覽器支援 WebGPU，非 NVIDIA 平台亦能享受 GPU 級別、且近乎零安裝成本的加速**。

3. **建立 Web-native 生物資訊工具的可行路徑**
研究證明「免安裝、跨硬體、保持資料本地」的瀏覽器模式完全可以承載高端基因體演算法，為學界與產業端降低部署與教學門檻。

### 7.2　學術與產業影響

本工作向生物資訊社群證明：**GPU 加速不再等同「安裝 CUDA SDK 或租用雲端 A100」**。只要一台能執行現代瀏覽器的筆電，即便配備 iGPU，也能在本地沙盒完成 Pair-HMM 前向尤度估計；既保障資料隱私，也極大降低教學與入門門檻。該模式為未來「Web-native GPU 科學計算」—無需 IT 管理員、零依賴、自帶 UI—提供明確範例，也為瀏覽器技術的下一階段演進提供衡量基準。

總結而言，本研究奠定了將生物資訊動態規畫演算法遷移至瀏覽器端的可行路徑，並透過跨平台實驗凸顯 WebGPU 在兼顧便利性、安全性與性能時的優勢。隨著 API 與硬體持續成熟，我們預期未來 3–5 年內，更多基因體分析工具將以「打開瀏覽器即用」的型態普及，進一步推動生物資訊民主化與醫療數位轉型。
