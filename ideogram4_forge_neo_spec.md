# Ideogram 4.0 — Forge Neo 統合 仕様書

**対象リポジトリ:** `Haoming02/sd-webui-forge-classic`（`neo` ブランチ）のフォーク
**追加対象モデル:** Ideogram 4.0（9.3B パラメータ open-weight text-to-image モデル）
**ライセンス区分:** Forge Neo 本体 = AGPL-3.0 / Ideogram 4.0 推論コード = Apache 2.0 / モデルウェイト = Ideogram Non-Commercial（商用は別途有料ライセンス）
**バージョン:** Draft 1.0

---

## 1. 概要

本仕様書は、Stable Diffusion WebUI Forge Neo に Ideogram 4.0 を新しいモデルタイプとして統合するための、UI 仕様とバックエンド仕様を定義する。

Ideogram 4.0 は既存の SD / SDXL / Flux とはアーキテクチャが根本的に異なる。具体的には次の 3 点が統合上の主要な相違点となる。

1. **flow-matching ベースの single-stream DiT**（34 層）であり、サンプラー選択や外部 LoRA・VAE・CLIP の差し替えという概念がない。
2. テキストエンコーダに **Qwen3-VL-8B-Instruct**（vision-language model）を固定で使用する。
3. 最大の品質を引き出すには **構造化 JSON キャプション**でプロンプトを与える必要があり、キーの順序が厳密に規定されている。

このため、既存 UI をそのまま流用するのではなく、モデルタイプを `Ideogram 4.0` に切り替えた際に専用の入力 UI と専用のバックエンドパイプラインへディスパッチする設計とする。

---

## 2. スコープと前提

### 2.1 スコープ内

- 画面左上のモデルタイプドロップダウンへの `Ideogram 4.0` 追加
- Ideogram 4.0 選択時の txt2img UI（JSON キャプションビルダーを含む）
- モデルローダー、JSON 組み立て、CaptionVerifier、推論パイプライン呼び出しを含むバックエンド処理
- nf4 / fp8 量子化ウェイトの読み込み

### 2.2 スコープ外（初期リリースでは対応しない）

- img2img / inpaint（Ideogram 4.0 は本リリースでは text-to-image のみ）
- 編集可能テキスト・レイヤー機能（Ideogram 側で "coming soon" とされている機能）
- LoRA 学習・適用
- ControlNet 連携

### 2.3 前提となる外部仕様（公式ドキュメント由来）

| 項目 | 値 |
| --- | --- |
| 対応解像度 | 高さ・幅ともに 16 の倍数、256〜2048 の範囲、アスペクト比 最大 6:1 |
| サンプラープリセット | `V4_QUALITY_48`（48 step）/ `V4_DEFAULT_20`（20 step）/ `V4_TURBO_12`（12 step） |
| ウェイト | `ideogram-ai/ideogram-4-nf4`（CUDA、Diffusers 対応）/ `ideogram-ai/ideogram-4-fp8`（全 HW、Diffusers 非対応） |
| ウェイト取得 | Hugging Face のゲート付き。ライセンス同意 + アクセストークン認証が必要 |
| JSON 検証 | `src/ideogram4/caption_verifier.py` の `CaptionVerifier` を流用 |

---

## 3. UI 仕様

### 3.1 モデルタイプドロップダウンへの追加

画面左上のモデルタイプ選択ドロップダウン（既存項目: `sd`, `xl`, `flux`, `Anima` 等）に新項目 `Ideogram 4.0` を追加する。

- **表示ラベル:** `Ideogram 4.0`
- **内部識別子:** `ideogram4`（バックエンドのディスパッチキーとして使用）
- **挙動:** 選択時、txt2img タブのプロンプト関連 UI を Ideogram 専用レイアウトに切り替える。同時にバックエンドのモデルローダー／パイプラインを `ideogram4` 系へディスパッチする。

> **補足:** Forge / Forge Neo の backend は本来 state-dict から `huggingface_guess` 相当の仕組みで自動判定を行うが、UI 側のモデルタイプ選択はその判定を上書き／補強する役割を持つ。Ideogram 4.0 については state-dict 構造が既存アーキテクチャと異なるため、UI 選択を「明示的なディスパッチトリガ」として扱う（詳細は §4.2）。

### 3.2 条件付き UI 表示ロジック

`Ideogram 4.0` 選択時、既存 UI の各要素は以下のように扱う。

| 既存 UI 要素 | 扱い | 理由 |
| --- | --- | --- |
| Checkpoint ドロップダウン | **流用** | Ideogram 4.0 のウェイトを選択 |
| Width / Height | **流用（範囲変更）** | 256〜2048・step 16・アスペクト比 6:1 制限を適用 |
| Steps | **流用（プリセット連動）** | プリセット選択時に自動設定 |
| CFG Scale | **流用（プリセット連動）** | 2 段階ガイダンススケジュールに対応 |
| Seed | **流用** | そのまま利用 |
| Batch count / size | **流用** | パイプライン側ループで対応 |
| Sampling Method | **非表示** | flow-matching 固定、サンプラー概念なし |
| LoRA / VAE / Text Encoder 選択 | **非表示** | 専用 VAE・Qwen3-VL 固定、外部差し替え不可 |
| Hires Fix / Upscaler | **非表示** | ネイティブ 2K 出力（後処理として残置は任意） |
| Negative Prompt | **改修** | Ideogram の dual-branch CFG の挙動に合わせる（§4.6 参照） |

### 3.3 推論パラメータパネル（新規・改修）

| ウィジェット | 種別 | 仕様 |
| --- | --- | --- |
| Sampler Preset | ドロップダウン（新規） | `V4_QUALITY_48` / `V4_DEFAULT_20` / `V4_TURBO_12`。選択時に Steps・CFG schedule・`mu`・`std` を自動反映。既定値は `V4_QUALITY_48` |
| `mu` | 数値入力（新規・上級者向け） | logit-normal スケジュール平均。プリセット連動。手動上書き可。解像度により自動調整 |
| `std` | 数値入力（新規・上級者向け） | logit-normal スケジュール標準偏差。プリセット連動。手動上書き可 |
| Transparent Background | チェックボックス（新規） | 透過背景出力（ロゴ・アイコン用途） |
| Resolution Preset | ドロップダウン（任意・新規） | 1024×1024 / 1536×1024 / 1024×1536 / 1920×1088 / 2048×768 / 1024×1792 / 1600×400 などの定型値を提供 |

`mu` / `std` はアコーディオン（"Advanced" 折りたたみ）内に配置し、通常はプリセットの自動設定に委ねる。

### 3.4 JSON キャプションビルダー（新規・本統合の中核）

プロンプト入力欄を、JSON キャプションの 3 つのトップレベル構造に対応した入力 UI に置き換える。各セクションはアコーディオンで折りたたみ可能とする。

#### 3.4.1 `high_level_description`（任意・強く推奨）

- **ウィジェット:** 複数行テキストエリア
- **内容:** 画像全体の 1〜2 文の要約
- **プレースホルダ例:** `A medium-shot photograph of a barista pouring latte art in a cozy cafe.`

#### 3.4.2 `style_description`（任意）

| サブフィールド | ウィジェット | 表示条件 | 備考 |
| --- | --- | --- | --- |
| `medium` | ドロップダウン | 常時 | `photograph` / `illustration` / `3d_render` / `painting` / `graphic_design` 等。値により `photo` / `art_style` 欄を切替 |
| `aesthetics` | テキスト入力 | 常時 | 美学キーワード（"moody, cinematic, desaturated"） |
| `lighting` | テキスト入力 | 常時 | 照明記述（"golden hour, rim light"） |
| `photo` | テキスト入力 | `medium == "photograph"` のとき | カメラ／レンズ詳細（"35mm, f/1.4, bokeh"） |
| `art_style` | テキスト入力 | `medium != "photograph"` のとき | アートスタイル記述（"flat vector illustration"） |
| `color_palette`（全体） | カラーピッカー × N | 常時 | **最大 16 色**。`#RRGGBB` 大文字。追加／削除ボタン付き |

> `photo` と `art_style` は排他。`medium` の選択に応じて一方のみ表示することで、ユーザーが両方を同時入力する誤りを UI レベルで防止する。

#### 3.4.3 `compositional_deconstruction`（必須）

| サブフィールド | ウィジェット | 備考 |
| --- | --- | --- |
| `background` | 複数行テキストエリア | 背景・環境の記述（必須） |
| `elements[]` | 動的リスト | 要素カードを追加／削除／並べ替え可能 |

各 `element` カードの内部構成:

| フィールド | ウィジェット | 表示条件 | 備考 |
| --- | --- | --- | --- |
| `type` | ラジオボタン | 常時 | `obj` / `text`。選択により下位フィールドを切替 |
| `bbox` | 数値入力 ×4 または矩形エディタ | 任意 | `[y_min, x_min, y_max, x_max]`、0〜1000 正規化座標（原点は左上）。ビジュアル矩形エディタが理想（§7 参照） |
| `text` | テキスト入力 | `type == "text"` のとき | 画像内にレンダリングするリテラル文字列 |
| `desc` | テキストエリア | 常時 | 要素の詳細記述 |
| `color_palette`（要素別） | カラーピッカー × N | 任意 | **最大 5 色**、`#RRGGBB` 大文字 |

### 3.5 補助機能（新規）

| 機能 | ウィジェット | 仕様 |
| --- | --- | --- |
| Plain-text mode | チェックボックス | ON でビルダー UI を非表示にし、従来型の単一プロンプト入力に切り替える。Ideogram はプレーンテキストも受理する |
| Magic Prompt | ボタン + 設定 | プレーンテキストを LLM で JSON へ展開。バックエンド設定: `ideogram-4-v1`（Ideogram ホスト API・無料）/ `claude-opus-v1` / `claude-sonnet-v1`（OpenRouter 経由）。API キー入力欄を設定タブに用意 |
| JSON Preview / Editor | 折りたたみテキストエリア | ビルダーから生成された JSON 全体を表示。手動編集を許可し、編集内容をビルダーへ反映（双方向は任意、最低限「生成 → 表示」を実装） |
| Validation 表示 | インラインメッセージ | CaptionVerifier の警告（未知キー・必須キー欠落・キー順序違反）を生成前にユーザーへ提示 |

---

## 4. バックエンド仕様

### 4.1 アーキテクチャ概要とデータフロー

```
[UI: モデルタイプ = ideogram4]
        │
        ▼
[ディスパッチャ] ──(type==ideogram4)──► [Ideogram4 パイプライン経路]
        │
        ▼
[JSON キャプション組み立て] ── §4.4
        │
        ▼
[CaptionVerifier 検証] ── §4.5（警告を UI へ返す。生成はブロックしない）
        │
        ▼
[Ideogram4 モデルローダー] ── §4.3（nf4 / fp8）
        │
        ▼
[Ideogram4Pipeline.__call__] ── §4.6（height/width/num_steps/guidance_schedule/mu/std/seed）
        │
        ▼
[（任意）Hive セーフティスクリーニング] ── §4.7
        │
        ▼
[画像出力 + PNG メタデータ書き込み]
```

### 4.2 モデルタイプ登録とディスパッチ

- **登録:** UI のモデルタイプ列挙に `ideogram4` を追加し、対応する処理経路を backend のディスパッチテーブルに登録する。
- **判定の優先順位:** Forge Neo の既存 state-dict 自動判定（`huggingface_guess` / `model_detection` 相当）は Ideogram 4.0 の重み構造を認識しない可能性が高い。したがって UI で `Ideogram 4.0` が明示選択された場合は、自動判定の結果よりも UI 選択を優先して `ideogram4` 経路へ強制ディスパッチする。
- **ファイル名規約（補助・任意）:** Forge Neo は Kontext / Qwen-Edit 等で「パスにキーワードを含めることで判定」する規約を持つ。これに倣い、Ideogram 4.0 のウェイトファイル／フォルダ名に `ideogram` を含めることを推奨規約とし、自動判定の補助に利用してもよい。

> **要確認（§7）:** `ideogram4` 経路の実際のフック箇所は、`backend/loader.py` の `forge_loader` / `split_state_dict` および `main_entry.py` のモデル選択ロジックである可能性が高い。フォーク後に実コードでフック箇所を確定すること。

### 4.3 モデルローダー

- **対応量子化:** `nf4`（CUDA 専用、Diffusers 対応）と `fp8`（全 HW、Diffusers 非対応）。UI の Checkpoint 選択または量子化選択に応じて切り替える。
- **ウェイト取得:** Hugging Face ゲート付きリポジトリ（`ideogram-ai/ideogram-4-nf4` / `ideogram-ai/ideogram-4-fp8`）。事前にライセンスゲートへの同意と HF アクセストークンが必要。未認証時は `404 / GatedRepoError` となるため、トークン未設定時は明示的なエラーメッセージを UI に返す（§4.8）。
  - トークンは設定タブまたは環境変数 `HF_TOKEN` から取得する。
- **テキストエンコーダ:** Qwen3-VL-8B-Instruct を固定でロードする。13 の中間層から hidden states を抽出して連結する処理は公式推論コードに準拠する。
- **VAE:** Ideogram 4.0 専用 VAE を使用。外部 VAE 差し替えは行わない。
- **メモリ管理:** Forge Neo 既存の `memory_management` を利用。nf4 版は 16GB VRAM クラスで動作する想定（公式コミュニティ報告に基づく）。

### 4.4 JSON キャプション組み立て（キー順序保証 — 最重要）

UI の各入力値から JSON 文字列を生成する。**キー順序がモデル品質に直結する**ため、組み立てロジックは以下の規約を厳守する。

#### 4.4.1 トップレベル順序

`high_level_description` → `style_description` → `compositional_deconstruction`

- `compositional_deconstruction` は必須。`background` → `elements` の順。
- `high_level_description`、`style_description` は任意（空なら省略）。

#### 4.4.2 `style_description` のキー順序

| キャプション種別 | 必須キー順序 |
| --- | --- |
| 写真（`photo` 使用） | `aesthetics` → `lighting` → `photo` → `medium` → `color_palette` |
| 非写真（`art_style` 使用） | `aesthetics` → `lighting` → `medium` → `art_style` → `color_palette` |

- `aesthetics` / `lighting` / `medium` は `style_description` 使用時は必須。
- `color_palette` は唯一省略可能なキーで、含める場合は必ず末尾。

#### 4.4.3 `elements[]` 各要素のキー順序

| type | 必須キー順序 |
| --- | --- |
| `obj` | `type` → `bbox` → `desc` → `color_palette` |
| `text` | `type` → `bbox` → `text` → `desc` → `color_palette` |

- `bbox`、`color_palette` は任意。含める場合は上記位置を維持。

#### 4.4.4 値のフォーマット規約

- **Hex カラー:** 大文字 `#RRGGBB` 形式のみ（`#1B1B2F` は可、`#1b1b2f` や `#fff` は不可）。UI のカラーピッカー出力を大文字 6 桁へ正規化する。
- **`bbox`:** `[y_min, x_min, y_max, x_max]`、整数、0〜1000。
- **シリアライズ:** Python `json` モジュール使用時は `separators=(",", ":")` かつ `ensure_ascii=False`。`\uXXXX` エスケープを避け、非 ASCII 文字はリテラルで保持する。

> **実装方針:** キー順序を Python の `dict` 挿入順に依存させるのではなく、明示的に「順序付きビルダー関数」を用意し、各セクションを規定順で組み立てる。これにより UI の入力順や省略の有無に関わらず順序が保証される。

### 4.5 CaptionVerifier 統合

- Ideogram 公式の `src/ideogram4/caption_verifier.py`（`CaptionVerifier`）をフォークに取り込み、生成前に組み立て済み JSON を検証する。
- 検出対象: 未知キー、必須キー欠落、キー順序違反、`color_palette` の形式不正、`\uXXXX` エスケープの混入。
- **挙動:** 検証はあくまで警告であり、生成をブロックしない（公式仕様上、スキーマ逸脱は許容されるため）。警告内容を UI のバリデーション表示（§3.5）へ返す。

### 4.6 推論パイプライン呼び出し

`Ideogram4Pipeline.__call__`（公式 `pipe(...)`）へ以下を渡す。

| パラメータ | 供給元 | 備考 |
| --- | --- | --- |
| prompt | §4.4 で組み立てた JSON 文字列（Plain-text mode 時は素の文字列） | — |
| `height` / `width` | UI スライダー | 16 の倍数・256〜2048 |
| `num_steps` | Sampler Preset | プリセット由来 |
| `guidance_schedule` | Sampler Preset | 2 段階スケジュール（本ステップ gw=7、ポリッシュ gw=3）。指定時は `guidance_scale` を上書き |
| `mu` / `std` | Sampler Preset または Advanced 手動値 | logit-normal スケジュール |
| `seed` | UI Seed | 再現性確保のため設定可 |

- **プリセット定義:** 公式 `ideogram4.PRESETS`（`src/ideogram4/sampler_configs.py`）を参照する。独自プリセット追加もこのレジストリへ行う。

| Preset | Steps | CFG schedule | `mu` | `std` |
| --- | --- | --- | --- | --- |
| `V4_QUALITY_48` | 48 | 45 step @ gw=7 + 3 polish step @ gw=3 | 0.0 | 1.5 |
| `V4_DEFAULT_20` | 20 | 18 step @ gw=7 + 2 polish step @ gw=3 | 0.0 | 1.75 |
| `V4_TURBO_12` | 12 | 11 step @ gw=7 + 1 polish step @ gw=3 | 0.5 | 1.75 |

- **dual-branch CFG とネガティブプロンプト:** Ideogram は conditional / unconditional の 2 分岐を独立に精緻化する。UI のネガティブプロンプトをどの分岐へ供給するかは公式推論コードの実装に準拠する（独自解釈で渡さない）。

### 4.7 セーフティスクリーニング（任意）

- 公式パイプラインは Hive によるプロンプト／出力のスクリーニングをオプションで備える（`HIVE_TEXT_MODERATION_KEY` / `HIVE_VISUAL_MODERATION_KEY`）。
- ローカル個人利用では必須ではないが、対応する場合は設定タブにキー入力欄を設ける。
- **注記:** Ideogram 4.0 はモデルウェイト内部に独自のセーフティフィルタを内蔵しており、これは外部から無効化・調整できない。ブロックや空出力が発生した場合はモデル内部フィルタの動作であり、バックエンドのバグとして扱わない。

### 4.8 エラーハンドリング

| 事象 | 対応 |
| --- | --- |
| HF トークン未設定／ゲート未同意（`GatedRepoError`） | 「ウェイト取得にはライセンス同意と HF トークンが必要」と明示し、設定箇所を案内 |
| 解像度がアスペクト比 6:1 超過 | 生成前にバリデーションし、許容範囲を提示 |
| 解像度が 16 の倍数でない | 自動的に直近の 16 の倍数へ丸める、または警告 |
| `nf4` を非 CUDA 環境で選択 | `fp8` への切替を促す |
| CaptionVerifier 警告 | 生成は継続しつつ UI に警告表示 |

---

## 5. データモデル（JSON キャプションスキーマ）

完全な参照例（写真・カラーパレット付き）:

```json
{
  "high_level_description": "A lone sailboat on calm water at sunset.",
  "style_description": {
    "aesthetics": "serene, warm, golden hour",
    "lighting": "golden hour backlighting, warm atmospheric haze",
    "photo": "wide angle, f/8, long exposure",
    "medium": "photograph",
    "color_palette": ["#FF6B35", "#F7C59F", "#004E89", "#1A659E", "#2B2D42"]
  },
  "compositional_deconstruction": {
    "background": "A calm ocean stretching to a low horizon, sky washed in orange and pink with thin wisps of cloud.",
    "elements": [
      {
        "type": "obj",
        "bbox": [200, 300, 800, 900],
        "desc": "A single sailboat with a white triangular sail, silhouetted against the setting sun."
      }
    ]
  }
}
```

グラフィックデザイン（非写真・テキスト要素付き）の例:

```json
{
  "high_level_description": "A clean, modern business card layout for a tech company.",
  "style_description": {
    "aesthetics": "minimal, professional, geometric",
    "lighting": "even, diffuse studio lighting",
    "medium": "graphic_design",
    "art_style": "flat vector design, generous whitespace, sans-serif typography",
    "color_palette": ["#FFFFFF", "#F0F0F0", "#333333", "#0066FF", "#00CC88"]
  },
  "compositional_deconstruction": {
    "background": "A solid off-white card surface with subtle paper texture.",
    "elements": [
      {
        "type": "text",
        "text": "ACME TECH",
        "desc": "Bold dark grey sans-serif company name across the upper third of the card."
      },
      {
        "type": "text",
        "text": "hello@acme.tech",
        "desc": "Small blue sans-serif contact email near the bottom of the card."
      }
    ]
  }
}
```

---

## 6. 実装フェーズ（推奨マイルストーン）

| フェーズ | 内容 | 完了条件 |
| --- | --- | --- |
| P0 | フォーク作成・`feature/ideogram4` ブランチ作成・upstream 設定 | ブランチが clone 可能 |
| P1 | バックエンド: ローダー + パイプライン最小実装（Plain-text mode のみ） | 単一プロンプトで画像生成が通る |
| P2 | UI: モデルタイプドロップダウン追加 + 条件付き表示ロジック | `Ideogram 4.0` 選択で専用 UI に切替 |
| P3 | JSON キャプションビルダー UI + 順序保証付き組み立てロジック | GUI 入力から正しい順序の JSON を生成 |
| P4 | CaptionVerifier 統合 + バリデーション表示 | 警告が UI に表示される |
| P5 | Sampler Preset 連動 + Advanced（mu/std）+ 透過背景 | プリセット選択で各値が自動反映 |
| P6 | Magic Prompt 連携 + JSON プレビュー/エディタ | プレーン文 → JSON 展開が動作 |
| P7（任意） | bbox ビジュアル矩形エディタ | キャンバス上でドラッグ配置可能 |
| P8（任意） | Hive セーフティスクリーニング | キー設定時にスクリーニング動作 |

---

## 7. 未確認事項・要確認リスト

フォーク後、実コードに対して以下を確定する必要がある。本仕様書では設計方針として記述しているが、Forge Neo 内部の正確な関数・ファイル構造は実装時に検証すること。

- [ ] モデルタイプドロップダウンの定義箇所と列挙の追加方法
- [ ] backend のディスパッチフック箇所（`backend/loader.py` の `forge_loader` / `split_state_dict`、`main_entry.py` のモデル選択ロジックが候補）
- [ ] state-dict 自動判定（`huggingface_guess` / `model_detection` 相当）が Ideogram 4.0 を誤判定しないか、UI 選択で確実に上書きできるか
- [ ] 既存 txt2img UI の条件付き表示・非表示を実現する Gradio コンポーネントの制御方法
- [ ] Qwen3-VL-8B エンコーダと既存メモリ管理（`memory_management`）の相性・VRAM 要件
- [ ] PNG メタデータ（infotext）へ JSON キャプションをどう保存・復元するか

---

## 8. ライセンス注記

- 本フォークは Forge Neo を継承するため **AGPL-3.0** で公開する必要がある（ネットワーク提供時のソース開示義務を含む）。
- Ideogram 4.0 推論コード（Apache 2.0）の取り込みは AGPL と両立する。
- **モデルウェイトは別ライセンス。** 研究・評価・個人プロジェクトは無料（Ideogram Open Model Agreement）だが、プロダクトや収益関連ワークフローでの使用は有料の商用ライセンスが必要。職場利用が商用に該当するかは Ideogram のライセンス条項で個別確認すること。
- ウェイトを再配布する場合は、Ideogram の Non-Commercial Agreement の写しと所定の attribution（`Notice` ファイル）の同梱が条件となる。**フォークのリポジトリにウェイトを同梱しない**（コードのみを配布し、ウェイトは各自が HF から取得する）構成が、ライセンス上もっとも単純で安全。

---

*本仕様書は公開情報（Ideogram 4.0 公式 GitHub / Forge Neo 公式 README およびソース）に基づくドラフトであり、実装時には対象コードベースでの検証を前提とする。*
