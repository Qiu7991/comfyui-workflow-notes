<div align="center">

# ComfyUI 工作流笔记 · 角色一致性 / 低显存 / 图生视频

**7 套可复用工作流的参数说明与踩坑记录**

`ComfyUI` · `Krea2` · `MiniMax H3` · `int8 量化` · `4-step Turbo LoRA`

</div>

---

## 这份笔记是什么

我把一条「AI 短片生产线」在消费级显卡上跑通之后，把每个工作流的**可用参数组合**和**踩过的坑**记了下来。

网上大量 ComfyUI 教程的问题是：只给 JSON 不给参数，或者只给「出图好看」的成品不给「为什么这么配」。等你换一个模型、换一个显存档位，那些参数全失效。

这份笔记反过来：**每个工作流都写清楚 UNET / VAE / LoRA / 步数 / CFG / 采样器 / 调度器 / 降噪，以及这些值为什么是这些值。**

---

## 目录

| 文件 | 用途 | 核心模型 |
|---|---|---|
| `workflows/01-krea2-txt2img-styles.json` | 文生图（角色定妆 / 场景 / 道具） | Krea2 int8 |
| `workflows/02-krea2-img2img.json` | 单图风格转换 | Krea2 int8 + identity LoRA |
| `workflows/03-krea2-image-edit.json` | 双图 / 单图编辑、扩展、局部重绘 | Krea2 int8 + identity LoRA |
| `workflows/04-h3-quadview.json` | 角色四视图 | Krea2 fp8 + QuadView LoRA |
| `workflows/05-h3-i2v-lowvram.json` | 图生视频（默认，低显存） | MiniMax H3 int8 + 4-step Turbo LoRA |
| `workflows/06-h3-accelerated-integrated.json` | 首尾帧 / 多参 / 文生视频 | MiniMax H3 int8 + Turbo LoRA |
| `workflows/07-experimental-accelerated.json` | 实验版（已知彩纹，不用于生产） | — |
| `docs/00-完整工作流手册.md` | 路由表 + 参数 + 交付格式 + 验收原则 | — |

---

## 三件事先记住

### 1. 画幅必须在生成前定死

这是最贵的一条教训。竖屏素材裁成横屏会切掉 58%–68% 画面，人物被拦腰切断——**这是几何问题，事后无解**。

抖音只有 9:16 能全屏铺满，4:3 和 16:9 必然留黑边。开工前先确认最终画幅。

### 2. 角色一致性靠参考图，不靠文字

旧做法是在提示词里写 "The same minimalist 2D animated stick figure..."，实测**完全无效**：9 条片段都写了 `no pom-pom`，模型照样画出毛球；脸型每条都不同。

正确做法：

```
生成角色立绘（C2A / Krea2 文生图）
→ 上传到视频模型的「角色」功能
→ 生成时附加角色缩略图
→ 提示词只写场景与动作，不写角色外观
```

### 3. 关键帧是锁定像素的唯一手段

文生视频每次重新采样，同一提示词两次结果都不同。可靠做法是用图像模型生成首尾帧（可控性远高于视频模型），喂给视频模型当首尾帧，角色像素被钉死。

**相邻两段共用关键帧**（Clip N 尾帧 = Clip N+1 首帧），既锁角色又保证衔接。

---

## 快速参数速查

### 文生图（角色 / 场景 / 道具）

```yaml
UNET:      Krea2\krea2_turbo_int8_convrot.safetensors
VAE:       qwen_image_vae.safetensors
比例:      9:16
步数:      8
CFG:       1
采样器:    euler
调度器:    simple
降噪:      1
风格选择器: 开启 → krea2_397styles-photography_摄影 → Photography
```

> ⚠️ JSON 里保存的是 `3d_render_3D渲染 + Backlit Portraiture`，**每次加载后必须手动改成 photography 组合**，否则风格完全不对。

开启风格模板后，主体提示词只写「人物或物体 + 年龄与必要外观 + 服装 + 动作 + 场景 + 构图」。不要重复写「写实摄影、电影感、曝光、景深、色调、皮肤纹理」——会和风格模板打架。

### 图生视频（低显存默认版）

```yaml
模式:      I2VA
UNET:      Minimax_H3\minimax_h3_fl2va_pruned_int8_convrot.safetensors
视频 VAE:  minimax_h3_video_vae_fp16.safetensors
音频 VAE:  minimax_h3_audio_vae_fp32.safetensors
LoRA:      minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors
LoRA强度:  0.75
采样器:    res_multistep
调度器:    simple
步数:      4
CFG:       1
首轮比例:  9:16
首轮倍率:  0.7
帧率:      24 fps
低显存补丁: SageAttention + ChunkFeedForward(4/4096) + LowVRAMAttention(10)
旁路:      TESpeedMiniMaxH3 / RTXVideoSuperResolution
```

首轮不启用超分。先检查人物、手部、道具、动作、声音和口型，通过后再决定是否提高倍率或后期放大。

---

## 低显存配置详解

这三重补丁解决的是三个不同的显存峰值，缺一个都会 OOM：

| 补丁 | 解决的问题 | 参数 |
|---|---|---|
| **SageAttention** | 注意力计算的中间张量峰值 | 启用 |
| **ChunkFeedForward** | FFN 层一次性加载全部通道 | `chunk=4`, `dim=4096` |
| **LowVRAMAttention** | 注意力分块执行，降低单块显存 | `chunk=10` |

配合 `int8_convrot` 量化（conv-rot 旋转量化，比朴素 int8 精度损失小）和 4-step Turbo LoRA，才能在消费级显卡上出片。

**为什么 LoRA 强度是 0.75 而不是 1.0**：4-step Turbo LoRA 本身会带来运动幅度衰减和细节扁平化，强度 1.0 时人物动作变得僵硬。0.75 是在「保持 4 步速度」和「保留运动自然度」之间的平衡点，实测这个值动作崩坏率最低。

---

## 提示词纪律

### 不要用比喻

写 `drains downward` → 画出一层水；改成 `evaporating like mist` → 画出一片雾；写 `dim abstract interior` → 画成一个盒子。

**描述光效和空间时只做平实陈述，不用修辞。**

### 多角色必须交代「谁是谁」

没写耳朵和尾巴的猴子 → 画成呆板人形；只写 `silhouette` 的剪影 → 画成主角的克隆。每个角色都要写明特征，并加负面约束防混淆。

### 动作要有因果，不能有隐形施力者

写「角色被抬到横线上方」——谁在抬？改成「地面下沉带走下方角色」，并明确「没有隐形力量」。

### 镜头物理要自检

镜子场景：角色面向镜头、镜中也是正面 → 几何错误。改成「角色背对镜头面对镜子」，镜中才映出正脸。

---

## 生成策略

### 单变量对照

- 一次只改一个变量（提示词 / 种子 / 参数），不同时改多个；
- 比较两种提示词路线时，保持首帧、时长、种子和采样参数不变；
- 同一问题换 2–3 个种子仍重复出现时，**停止抽卡**，改提示词、拆镜头或换分组。

### 先验单张，再批量

```
单张资产 → 人工确认 → 关键帧 → 人工验收 → 批量生成视频 → 质检
```

不要在角色图未确认时做四视图，不要在关键帧未通过时批量生成视频。

### 每段一个主要动作

复杂动作拆镜头。10 秒是模型的生成容器，不是叙事单位。

---

## 交付格式

每个任务必须记录：

```text
任务类型:    生图 / 图片编辑 / 四视图 / 图生视频 / 首尾帧 / 多参考视频
工作流文件:  具体 JSON 文件名
启用分组:    具体分组名称
输入素材:    图像-1 / 图像-2 / 首帧分别加载什么
风格选择器:  开启/关闭；分类；具体风格
模型与采样:  UNET / LoRA / 步数 / CFG / 采样器 / 调度器 / 倍率
提示词:      可直接粘贴的最终内容
时长:        视频任务的具体秒数
保存命名:    文件名前缀
验收重点:    本轮只检查哪些问题
```

---

## 已知限制

- 工作流基于特定模型版本（Krea2 int8_convrot / MiniMax H3 fl2va_pruned_int8），换模型需重新调参
- 07 实验版已知会生成彩纹，仅用于对照测试
- 参数是在单张消费级显卡上调出来的，不同显存档位需要重新验证
- 不含任何模型权重文件，需自行下载

---

## License

本笔记与工作流 JSON 以 MIT 发布。模型权重与第三方节点的许可各自独立，请遵守上游协议。
