# Desktop2Stereo 跨平台授权客户端与启动器实施计划

## 1. 文档范围

本文只负责 `desktop2stereo-vulkan` 客户端：Windows、Linux、macOS 启动器，设备身份，
安全存储，授权选择，离线凭证验证，在线租约，Runtime 二次门禁，以及三平台打包发布。

服务器端账号、授权状态机、订单、支付、余额、邀请、提现、管理后台和部署由
`desktop2stereo-site/docs/13-cross-platform-licensing-server-implementation-plan.md` 负责；
服务器部署以 `desktop2stereo-site/docs/d2s.site.md` 为唯一基准。本文不再维护服务器表结构、
支付实现或腾讯云/Cloudflare 部署步骤。

截至 2026-09-04，Windows 启动器已有基础实现，但仍需迁移到 `desktop2stereo-site` 的
`/api/v1` 契约并重新验收；Linux x86_64、macOS arm64 正式启动器、生产公钥和三平台发布
流水线尚未完成。全部验收门通过前，不得把客户端标记为生产可用。

## 2. 客户端目标与边界

- Windows x86_64、Linux x86_64、macOS arm64 提供一致的登录、授权选择和启动体验。
- 支持设备码登录、一个账号多份授权、在线、7/14/30 天离线和永久绑定模式。
- 离线时只信任服务端 ES256 签名凭证和本地可信时间，不信任本地可编辑配置。
- 在线模式持有短期租约并定期续期；租约到期后安全停止受保护 Runtime。
- 启动器和 Runtime 各执行一次授权检查，不能通过直接运行 Python/EXE 绕过。
- 凭证失效时明确阻止受保护功能并给出可操作错误，不使用毒化模型或错误画面。
- 客户端不计算可信价格、不确认支付成功、不修改授权状态、不包含服务端私钥或支付 Secret。

## 3. 组件边界

### 3.1 共享授权核心

在 `src/desktop2stereo/auth/` 维护无 UI 的共享模块：

- API 客户端与稳定错误码映射。
- 设备指纹标准化与摘要。
- Device Authorization Grant 轮询状态机。
- Access Token 内存管理、Refresh Token 安全存储和刷新。
- 公钥清单、ES256 JWS 验证、授权选择与本地可信时间。
- 在线租约申请、续期、退出和到期通知。
- Runtime 门禁结果与诊断事件。

共享核心不能直接依赖 Flet 页面，也不能在日志中输出访问令牌、刷新令牌、完整凭证、设备
原始标识或个人支付资料。

### 3.2 启动器

启动器负责登录 UI、授权列表、设备/模式操作、购买入口、安全存储、单实例锁、Runtime
子进程启动和退出清理。启动器只管理自己创建的进程，不搜索或结束系统中的其它 Python
进程。

### 3.3 Runtime 二次门禁

`app_runtime.bootstrap` 和实际受保护运行入口必须重新读取授权上下文并验证：产品、授权 ID、
设备摘要、模式、签名、有效期和在线租约。只设置环境变量、命令行参数、标志文件或篡改
`settings.yaml` 不能绕过门禁。

## 4. 服务器 API 对接

权威契约为 `desktop2stereo-site/docs/desktop2stereo-api.md`，基础地址生产使用
`https://d2s.site/api/v1`。客户端至少对接：

- `POST /device/authorize`、`/device/token`、`/device/cancel`。
- `GET /license/list`、`/license/status`、`/license/keys`。
- `POST /license/activate`、`/switch`、`/change-mode`、`/renew`。
- `POST /license/offline/issue`、`/offline/extend`。
- `POST /license/revoke/free`、`/revoke/paid`。
- `POST /license/online/heartbeat`、`/online/logout`。
- `POST /license/permanent/confirm`、`/license/manual-unbind`。
- 需要打开网页时使用 d2s.site 的订单、邀请、余额和提现页面，不在客户端复制收银台。

所有响应按 `version`、`success`、`request_id` 和稳定 `error.code` 解析。未知版本或未知关键
字段必须安全失败，并显示 `request_id` 供客服排查。网络超时、401、403、409、429 和 5xx
必须分别处理，不能都显示“网络错误”。

## 5. 设备身份

各平台只采集稳定、非用户可编辑的系统标识，标准化后加入固定产品域分隔符并使用 SHA-256：

- Windows：优先系统 MachineGuid，使用受限 API 读取。
- Linux：优先 `/etc/machine-id` 或 `/var/lib/dbus/machine-id`。
- macOS：优先 IOPlatformUUID。

上传内容仅为 64 字符小写十六进制摘要和 `fingerprint_version`。原始标识不得上传、写日志或
写入普通配置。读取失败时显示诊断并阻止永久绑定；不得随机生成一个会在重启后变化的设备
身份冒充稳定指纹。

指纹算法升级必须支持版本并存和受控迁移，不能让旧版本客户端静默占用新设备授权。

## 6. 登录与令牌存储

设备码流程：

1. 启动器请求设备码，显示 `user_code`、验证网址和二维码。
2. 打开系统浏览器，用户在网页完成登录、邮箱验证和批准。
3. 启动器按服务端 `interval` 轮询；处理 pending、slow_down、expired、denied 和成功。
4. 成功后 Access Token 仅留内存，Refresh Token 写入平台安全存储。
5. 退出时调用授权在线退出和账号退出，清除安全存储、内存令牌与本地会话缓存。

平台安全存储：

- Windows：Credential Manager 或 DPAPI。
- macOS：Keychain。
- Linux：Secret Service/libsecret；服务不可用时只提供本次会话登录，不将 Refresh Token
  降级保存到明文文件。

日志、崩溃转储和遥测必须对 Authorization、Cookie、JWS 和设备摘要做脱敏。

## 7. 离线凭证验证

发布包内置“当前键 + 上一键”公钥清单，按 `key_id` 选择 P-256 公钥验证 ES256 紧凑 JWS。
必须验证：

- 算法固定为 ES256，拒绝 `none`、算法替换和未知键。
- `version`、`product=desktop2stereo`、`license_id`、`device_hash` 和模式匹配。
- `not_before <= trusted_now < expires_at`，并对边界条件做自动化测试。
- 试用、离线天数和 features 只能来自已验证 claims。

客户端记录最近服务端签名时间和最高可信本地时间。时钟明显回拨时进入 `clock_suspect`，只允许
联网校时和重新授权；不能通过删除普通缓存恢复离线运行。凭证缓存应使用平台安全存储或受
完整性保护的本地数据，删除缓存只会要求重新联网，不会产生新权限。

## 8. 在线租约与运行期行为

- Runtime 启动前申请租约；租约有效期 15 分钟，正常每 5 分钟续期。
- 同一授权被占用时显示占用状态和处理建议，不自动踢掉另一运行实例。
- 临时断网可运行到已签发租约截止；截止前持续重试并采用有上限的退避。
- 到期后停止新帧提交、保存可安全保存的本地设置并退出受保护 Runtime。
- 正常退出调用 `/license/online/logout`；异常退出依靠短租约自然回收。
- 系统睡眠/唤醒、网络切换和本地时钟变化后立即重新确认租约。

在线租约令牌只驻留当前运行会话，不写入 `settings.yaml` 或长期安全存储。

## 9. 三平台启动器与打包

### Windows x86_64

- 正式入口：`Desktop2Stereo.exe`，无控制台窗口。
- 使用 Authenticode 签名；安装包、卸载、升级和 SmartScreen 行为需要实机测试。
- 准备完成后写 `auth_ready.flag`，Runtime GUI 完成后写 `gui_ready.flag`。

### Linux x86_64

- 正式入口：`Desktop2Stereo-x86_64.AppImage`。
- 对 Ubuntu LTS 和至少一个常用发行版验证 Secret Service、Wayland/X11、Vulkan 和桌面入口。
- 无 Secret Service 时不持久化登录，不要求用户关闭系统密钥服务安全策略。

### macOS arm64

- 当前目录契约中的正式入口为 `src/Desktop2Stereo-macos`；未来 DMG 只包装独立启动器。
- 使用 Developer ID 签名并公证，验证 Gatekeeper、Keychain、Apple Silicon 和 Vulkan/MoltenVK
  相关依赖。
- 不依赖用户手动执行 `xattr` 绕过正式发布安全检查。

BAT、Bash、`run_mac` 和 Python 入口只用于开发/诊断，不进入正式用户发布包。

## 10. 错误码与用户体验

至少提供下列可操作状态：未登录、邮箱未验证、设备码待批准/过期、无授权、授权到期、设备
不匹配、授权被占用、撤销冷却、永久绑定锁定、时钟异常、签名/公钥异常、服务器限流、服务
暂不可用和安全存储不可用。

错误页保留“重试”“重新登录”“打开 d2s.site”“复制 request_id”“退出”中的适用操作；
不得无限轮询、自动重复下单或隐藏永久绑定不可逆提示。

## 11. 实施阶段与验收门

### C1：API 契约迁移

状态：`in_progress`。

- 将旧 `desktop2stereo-server` 地址和响应解析迁到 `desktop2stereo-site /api/v1`。
- 对齐设备码、授权列表、稳定错误码、公钥、离线 claims 和租约。
- 使用模拟服务器覆盖正常、超时、限流、401、409、5xx 和未知版本。

### C2：共享授权核心与 Runtime 门禁

状态：已有基础实现，需按新契约重验。

- 完成设备指纹、安全存储、JWS、可信时间、租约和退出清理。
- `bootstrap` 与运行入口二次检查，不接受单纯环境变量或标志文件授权。
- 覆盖凭证篡改、错误设备、过期、回拨、未知键和直接入口绕过测试。

### C3：Linux 与 macOS 启动器

状态：`planned`。

- 抽离共享核心，完成 Linux x86_64 和 macOS arm64 原生外壳与安全存储适配。
- 三平台保持相同业务状态机，平台差异只位于设备、安全存储和进程管理适配层。

### C4：发布流水线

状态：`planned`。

- 锁定依赖、生成 SBOM、构建可追溯产物并执行恶意软件/Secret 扫描。
- Windows 签名、macOS 签名公证、Linux AppImage 构建均在隔离发布环境完成。
- CI 缺少公钥、包含私钥、键版本不匹配或测试失败时必须阻止发布。

### C5：端到端与实机验收

状态：`planned`。

- 在三种真实系统对接生产等价服务器，覆盖注册批准、试用、购买、绑定、模式、撤销、租约、
  离线、退出、升级和卸载。
- 覆盖断网、睡眠唤醒、代理、服务端故障、并发运行、时钟回拨和安全存储不可用。
- 启动器显示版本、Git SHA、服务器环境和可脱敏导出的诊断信息。

只有 C1-C5 全部达到 `verified` 且服务器端上线门禁同时通过，才允许正式发布。

## 12. 自动化与实机测试清单

- 单元：设备摘要、错误映射、JWS、时间边界、租约退避、授权选择和状态转换。
- 集成：设备码单次消费、令牌刷新/重放、离线签发、在线冲突、撤销和退出清理。
- 安全：篡改 JWS、替换公钥、未知算法、明文 Secret 扫描、直接 Runtime 入口和参数注入。
- 打包：正式产物不包含开发入口、私钥、测试令牌、`.env` 或用户本地设置。
- 实机：Windows 11 x86_64、Linux x86_64、macOS arm64 的安装、首次运行、升级和卸载。

测试证据、当前实现路径和状态同步维护在 `docs/requirements-matrix.md`。

## 13. 当前待决策

- 确定 Linux 首发支持的发行版与 Secret Service 不可用时的产品提示。
- 确定 macOS 首发包装形式和 Developer ID 公证账号。
- 确定生产 P-256 公钥、轮换窗口和客户端旧键保留周期。
- 确定租约到期时 Runtime 的安全退出体验及可保存数据范围。

## 14. 文档同步规则

- 客户端行为或契约变化时同步本文、`docs/requirements-matrix.md` 和 `changelog.md`。
- 服务端 API 变化先更新 `desktop2stereo-site/docs/desktop2stereo-api.md`，再更新客户端适配。
- 部署变化只更新 `desktop2stereo-site/docs/d2s.site.md`，不复制到本文。
- `src/desktop2stereo/settings.yaml` 不保存密码、令牌、设备原始标识、区域、余额、支付资料或私钥。
