# Desktop2Stereo 跨平台授权启动器生产上线补充计划

本文是
[`13-cross-platform-licensing-launcher-implementation-plan.md`](./13-cross-platform-licensing-launcher-implementation-plan.md)
的生产落地补充，不替代原实施计划。原计划定义完整功能和长期架构，本文定义当前优先级、生产凭据、部署顺序和发布签名操作。

## 1. 两阶段原则

### 阶段一：先上线授权登录

第一阶段只开放并稳定验证以下流程：

1. 注册
2. 邮箱验证
3. 登录
4. 授权状态验证
5. 设备绑定
6. 退出登录

支付、邀请奖励、余额、提现以及正式发布签名不作为阶段一的上线阻塞项。

### 阶段二：支付与三平台签名

阶段一稳定后，再依次接入：

1. 支付 Sandbox
2. 支付生产 Webhook
3. Windows Authenticode 签名
4. macOS Developer ID 签名与公证
5. Linux 发布清单签名
6. Windows、Linux、macOS 实机验收

## 2. Cloudflare 生产授权配置

### 2.1 Turnstile

在 `desktop2stereo-server` 目录执行：

```powershell
npx wrangler turnstile widget create d2s-login --domain d2s.site --mode managed
```

如果生产站点使用 `www.d2s.site`，创建组件时同时加入该域名。

命令返回的 Site Key 写入 `wrangler.json` 的 `TURNSTILE_SITE_KEY`；Secret Key 只写入 Cloudflare Worker Secret：

```powershell
npx wrangler secret put TURNSTILE_SECRET_KEY
```

Site Key 可以公开，Secret Key 不得提交 Git。服务端通过 Cloudflare Siteverify 接口验证令牌。

参考：[Cloudflare Turnstile 服务端验证](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/)

### 2.2 腾讯云 SES

当前项目使用腾讯云 SES，不再使用旧的 Bearer 邮件服务配置。生产凭据必须通过 Worker Secret 注入：

```powershell
npx wrangler secret put TENCENTCLOUD_SECRET_ID
npx wrangler secret put TENCENTCLOUD_SECRET_KEY
```

当前已确定的非敏感配置：

```text
TENCENTCLOUD_REGION=ap-hongkong
TENCENTCLOUD_FROM_EMAIL=noreply@d2s.site
TENCENTCLOUD_VERIFY_TEMPLATE_ID_ZH=214121
TENCENTCLOUD_RESET_TEMPLATE_ID_ZH=214122
TENCENTCLOUD_VERIFY_TEMPLATE_ID_EN=214123
TENCENTCLOUD_RESET_TEMPLATE_ID_EN=214124
```

四个模板必须审核通过后才能进行真实发送。模板正文使用单一变量 `{{token}}`，并将链接固定写成
`https://d2s.site/verify-email?token={{token}}` 或
`https://d2s.site/reset-password?token={{token}}`；服务端根据请求的 `Accept-Language` 选择中文或英文模板，并通过腾讯云 SES `SendEmail` 的 TC3-HMAC-SHA256 签名请求发送。

腾讯云地域、模板 ID、发信地址可以作为 `wrangler.json` 的 `vars`；`SecretId` 和 `SecretKey` 不得写入 `wrangler.json`、源码、GitHub 普通变量或聊天记录。

参考：[腾讯云 SES 签名方法](https://cloud.tencent.com/document/product/1288/51058)、[腾讯云 SES 发送邮件](https://cloud.tencent.com/document/product/1288/51034)、[腾讯云 SES 发信模板](https://cloud.tencent.com/document/product/1288/55193)

### 2.3 管理员账号

完成一次真实注册和邮箱验证后，从账号信息中取得用户 ID，再将其作为受保护生产变量配置：

```json
{
  "vars": {
    "ADMIN_USER_IDS": "用户ID1,用户ID2"
  }
}
```

管理员 ID 不是密码，但应限制修改权限并记录变更。

## 3. 阶段一部署和验收

部署前检查：

```powershell
npx wrangler secret list
npx wrangler deploy
npx wrangler tail
```

验收至少覆盖：

- 注册成功后收到邮箱验证邮件。
- 中文和英文 `Accept-Language` 分别使用对应模板。
- 验证链接成功、重复使用失败、过期失败。
- 已验证账号可以登录，未验证账号不能登录。
- 设备码授权、轮询、绑定和退出登录正常。
- 错误请求、重复请求和网络失败不会伪造授权成功。
- 日志不输出 `SecretKey`、访问令牌、刷新令牌或完整邮箱验证令牌。

当前开发构建中的 `D2S_SKIP_AUTH=1` 仅用于本地调试。生产发布前必须从 Python 启动路径和 Windows 原生启动器中移除该开关，并重新构建、测试和打包。

## 4. 支付 Sandbox 与生产 Webhook

只有阶段一稳定后才配置支付凭据。每个平台的 Webhook Secret 使用 Cloudflare Worker Secret 保存：

```powershell
npx wrangler secret put PAYMENTFM_WEBHOOK_SECRET
npx wrangler secret put CREEM_WEBHOOK_SECRET
npx wrangler secret put PAYPAL_WEBHOOK_SECRET
npx wrangler secret put PADDLE_WEBHOOK_SECRET
npx wrangler secret put STRIPE_WEBHOOK_SECRET
```

各平台后台的 Webhook 地址必须指向项目对应的生产接口，例如：

```text
https://d2s.site/api/v1/webhooks/stripe
```

先使用 Sandbox 覆盖以下场景：

- 正常支付。
- 重复 Webhook。
- 签名错误。
- 退款或撤销。
- 网络重试和乱序到达。

服务端必须验证签名、订单归属、金额、币种和幂等事件 ID；客户端不得自行判定支付成功。支付 Webhook Secret 不得放入 `wrangler.json`、源码或 GitHub 普通变量。

## 5. 三平台发布签名

### 5.1 Windows Authenticode

正式发布需要 Authenticode 代码签名证书，准备 `.pfx`、PFX 密码和可信 RFC3161 时间戳服务地址。签名使用 SHA-256 和可信时间戳：

```powershell
signtool sign `
  /fd SHA256 `
  /f certificate.pfx `
  /p "$env:PFX_PASSWORD" `
  /tr "时间戳服务地址" `
  /td SHA256 `
  Desktop2Stereo.exe
```

验证：

```powershell
signtool verify /pa /v Desktop2Stereo.exe
```

证书文件和密码只允许在 GitHub `production` Environment 或受控发布机中使用。

参考：[Microsoft SignTool](https://learn.microsoft.com/en-us/windows/win32/seccrypto/using-signtool-to-sign-a-file)、[Authenticode 时间戳](https://learn.microsoft.com/en-us/windows/win32/seccrypto/time-stamping-authenticode-signatures)

### 5.2 macOS 签名和公证

准备 Apple Developer ID Application 证书、`.p12`、密码、Team ID，以及 App Store Connect API Key 或 `notarytool` 凭据。

项目继续使用位于发布包 `src/` 的独立 `Desktop2Stereo-macos` 可执行文件，不恢复 `.app` 文件夹：

```bash
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: YOUR_NAME" \
  Desktop2Stereo-macos
```

公证：

```bash
xcrun notarytool submit Desktop2Stereo-macos.zip \
  --key AuthKey.p8 \
  --key-id KEY_ID \
  --issuer ISSUER_ID \
  --wait
```

验证：

```bash
codesign --verify --deep --strict Desktop2Stereo-macos
spctl --assess --type execute --verbose Desktop2Stereo-macos
```

参考：[Apple Developer ID](https://developer.apple.com/help/account/certificates/create-developer-id-certificates)、[Apple 公证](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)

### 5.3 Linux 发布签名

Linux 优先签署发布清单，不引入 `.app` 结构：

```text
Desktop2Stereo-linux
release-manifest.json
release-manifest.json.sig
```

推荐 GitHub Actions 使用 OIDC + Sigstore/cosign。若暂时使用 GPG 或 cosign 私钥，私钥必须放入 GitHub `production` Environment Secret。

## 6. GitHub Actions 凭据和发布环境

创建 GitHub Environment：

```text
production
```

为该环境配置审批人，并仅向正式发布 Job 暴露以下凭据：

- Windows PFX Base64 和 PFX Password。
- Apple `.p12`、`.p12` Password 和公证 API Key 信息。
- Linux 发布签名凭据。

当前 `build-native-launcher.yml` 尚未完成 Windows 签名、macOS 公证和 Linux 发布签名集成；添加 Secret 本身不会产生签名，后续必须将签名步骤接入 Workflow，并在签名后执行验证。

## 7. 推荐执行清单

1. 创建并验证 Turnstile。
2. 确认腾讯云 SES 域名、发信地址和四个模板审核通过。
3. 配置 Cloudflare Worker Secrets 和生产 `vars`。
4. 移除开发免验证开关，部署并验证注册、邮箱验证、登录和设备授权。
5. 配置管理员 ID。
6. 使用支付 Sandbox 测试订单和 Webhook 幂等性。
7. 配置支付生产 Webhook。
8. 接入 Windows 签名。
9. 接入 macOS 签名和公证。
10. 接入 Linux 发布签名。
11. 完成三个平台的真实安装、登录、授权选择和 Runtime 验收。

所有 Secret、私钥、PFX、`.p12` 和 `.p8` 内容都不得发送到聊天或提交到仓库。
