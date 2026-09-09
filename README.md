# WanCare Connect

Build a complete mobile-responsive web application for a Japan-centric 
AI-powered smart dog collar app. This is for a Japanese hackathon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎨 DESIGN SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Colors:
- Primary: #1A2F5A (Deep Navy)
- Secondary: #FFB7C5 (Sakura Pink)
- Success: #4CAF82 (Mint Green)
- Warning: #F4A623 (Gold)
- Danger: #E53935 (Emergency Red)
- Background: #FAFAFA
- Cards: #FFFFFF
- Text: #1A1A2E

Typography:
- Google Font: Noto Sans JP (Japanese)
- Google Font: Inter (English)
- All labels bilingual: Japanese primary, English secondary

Style:
- Japanese mobile app aesthetic (like LINE / SmartNews)
- Kawaii but professional
- Rounded cards (border-radius: 16px)
- Soft shadows: 0 2px 12px rgba(0,0,0,0.08)
- Information-dense layout (Japanese users prefer this)
- Minimum 44px touch targets (elderly accessibility)
- Bottom navigation bar (mobile style)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📱 PAGES & SCREENS TO BUILD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Build these pages with React + Tailwind CSS:

―――――――――――――――――――――――――――――――――――
PAGE 1: SPLASH SCREEN
―――――――――――――――――――――――――――――――――――
- Centered app logo (dog silhouette + tech collar + wifi signal)
- App name: "WanCare / ワンケア" 
- Animated paw print loading dots
- Soft sakura pink gradient background
- Auto-redirects to Language Selection after 2 seconds

―――――――――――――――――――――――――――――――――――
PAGE 2: LANGUAGE SELECTION
―――――――――――――――――――――――――――――――――――
- Title: "言語を選択 / Select Language"
- 3 selectable cards:
  → 🇬🇧 English Only
  → 🇯🇵 日本語 Japanese Only
  → 🌐 Mixed (English + Japanese)
- Selected card: sakura pink border + light pink fill
- "続ける / Continue" full-width navy button at bottom
- Subtle paw print pattern in background

―――――――――――――――――――――――――――――――――――
PAGE 3: LOGIN / SIGNUP
―――――――――――――――――――――――――――――――――――
Login Tab:
- "おかえりなさい / Welcome Back" heading
- Email input field
- Password input with show/hide toggle
- "ログイン / Login" navy full-width button
- "パスワードを忘れた / Forgot Password?" link
- OR divider
- Social buttons: Google, Apple, LINE (LINE most prominent)
- Switch to Signup link

Signup Tab:
- "はじめまして / Nice to Meet You" heading
- Fields: Name, Email, Password, Confirm Password
- LINE signup button (prominent)
- Terms checkbox
- "登録 / Register" CTA button

―――――――――――――――――――――――――――――――――――
PAGE 4: ONBOARDING - STEP 1 (Avatar Setup)
―――――――――――――――――――――――――――――――――――
- Paw print step progress indicator: ●○○
- Title: "ワンちゃんを作ろう / Create Your Dog"
- Large circular avatar preview (120px)
- Breed selector as horizontal scrollable chips:
  柴犬 Shiba Inu | トイプードル Toy Poodle | 
  チワワ Chihuahua | ポメラニアン Pomeranian | 
  ゴールデンレトリバー Golden | ミックス犬 Mixed | その他 Others
- When "ミックス犬" selected: searchable dropdown appears
- Avatar customization panel:
  → Fur color swatches (6 colors)
  → Ear shape options (4 styles shown as small icons)
  → Eye style options (kawaii)
  → Collar color picker
- "次へ / Next" button

―――――――――――――――――――――――――――――――――――
PAGE 5: ONBOARDING - STEP 2 (Dog Details)
―――――――――――――――――――――――――――――――――――
- Paw print progress: ●●○
- Photo upload circle (dashed border, camera icon)
- Form fields (card-based):
  → 名前 Name* (required)
  → 年齢 Age (optional)
  → 体重 Weight in kg (optional)
  → 犬種 Breed (auto-filled, editable)
  → 性別 Gender toggle: オス Male / メス Female
  → ワクチン Vaccinated: Yes/No toggle
- Required fields marked with red asterisk
- Optional fields labeled in gray "任意 / Optional"
- "次へ / Next" button

―――――――――――――――――――――――――――――――――――
PAGE 6: ONBOARDING - STEP 3 (Owner Details)
―――――――――――――――――――――――――――――――――――
- Paw print progress: ●●●
- Simple kawaii illustration: person with dog
- Fields:
  → オーナー名 Your Name* (required)
  → 年齢 Age (optional)
  → 都道府県 Prefecture — dropdown of all 47 Japan prefectures
- "完了 / Complete Setup" large navy CTA button

―――――――――――――――――――――――――――――――――――
PAGE 7: HOMEPAGE / DASHBOARD (Main Screen)
―――――――――――――――――――――――――――――――――――
Top Bar:
- Left: Profile avatar (circular 40px)
- Center: "こんにちは、[Dog Name]! 🐾"
- Right: Notification bell + RED SOS button (pulsing animation)

SOS Button:
- Bright red pill button "🆘 SOS 緊急"
- Pulse/glow animation
- On click: shows emergency call modal

Hero Profile Card:
- Dog avatar with green status ring (collar connected)
- Dog name + breed + age chips
- "プロフィール編集 / Edit Profile" button
- "+ ペットを追加 / Add Pet" secondary button
- Collar status: green dot "接続済み / Connected"

Daily Insight Card:
- Light yellow background
- 🐾 icon
- "今日の豆知識 / Daily Dog Fact"
- Fact text that changes every 10 seconds (use array of 5 sample facts)
- Fact counter "#1,247"

Overall Health Score Card:
- Large circular progress ring (like Apple Watch)
- Score: 85/100
- Color: green (good), yellow (warning), red (danger)
- "総合健康スコア / Overall Health Score"
- Pulsing green dot "● LIVE"

Collar Connect Card:
- Connection status
- Battery level: 78% with battery icon
- Signal strength: 3 bars
- Last sync: "2 minutes ago / 2分前"
- "カラーを接続 / Connect Collar" button (if disconnected state)

AI Sensors Grid (2-column card grid):
Each card has: colored icon, sensor name JP+EN, 
live value, status badge (良好/Good, 注意/Warning, 危険/Danger)

① 🐶 BarkSense AI / 吠え分析
   Value: "穏やか / Calm"
   Badge: 良好 (green)
   Small: "ML学習中 / Learning your dog"

② 🔬 SkinSense AI / 皮膚センサー
   Value: "正常 / Normal"  
   Badge: 良好 (green)
   Small: "ML学習中 / Learning your dog"

③ 🏃 MotionSense AI / 運動センサー
   Value: "2,340歩 / Steps"
   Badge: 良好 (green)
   Progress bar showing daily goal

④ 🌡️ TemperatureSense AI / 体温センサー
   Value: "38.5°C"
   Badge: 良好 (green)
   Normal range indicator

⑤ 📍 LocationSense AI / 位置センサー
   Value: "渋谷区, 東京"
   Badge: 追跡中 (tracking, blue)
   Mini map preview thumbnail

⑥ 💨 PressureSense AI / 圧力センサー
   Value: "正常範囲 / Normal"
   Badge: 良好 (green)

⑦ 💡 LightSense AI / 光センサー
   Value: "室内 / Indoor"
   Badge: 良好 (green)
   RGB indicator dots

⑧ 🔗 CombineSense AI / 総合分析
   Value: "総合スコア / Combined"
   Badge: 良好 (green)
   Links to full report

Bottom Navigation Bar (fixed):
🏠 ホーム | 🗺️ 地図 | 🤖 AI | 🏥 クリニック | 👥 コミュニティ

―――――――――――――――――――――――――――――――――――
PAGE 8: MAP / LOCATION SCREEN
―――――――――――――――――――――――――――――――――――
- Full screen map placeholder (gray map background with grid)
- Custom paw print marker for dog location
- Blue dot for owner location
- Distance card: "🐾 あなたの犬 / Your Dog — 0.3km away"

Bottom Sheet (draggable panel):
- Dog name + current address in Japanese
- Last updated: "今たった / Just now"
- "道案内 / Get Directions" → Google Maps button
- Safe Zone toggle with radius circle
- "🔴 迷子モード / Lost Mode" toggle

When Lost Mode ON:
- Red emergency banner appears at top
- "24時間獣医を検索中 / Finding 24hr Vet..."
- Nearest vet card appears on map
- "今すぐ電話 / Call Now" red button
- "獣医にアラート送信済み / Alert Sent to Vet" green confirmation

―――――――――――――――――――――――――――――――――――
PAGE 9: CLINICS SCREEN
―――――――――――――――――――――――――――――――――――
Top: Search bar "クリニックを検索 / Search Clinics" + Filter button

Category tabs (horizontal scroll):
⭐ 高評価 Top Rated | 📍 近く Nearby | 
👍 おすすめ Recommended | 🔬 専門 Specialized | 
🏛️ 公立/私立 Public/Private

Filter Bottom Sheet (when filter clicked):
- Distance slider (0-10km)
- Rating filter (star selector)
- Open Now toggle
- 24hr Emergency toggle
- Specialization checkboxes

Clinic Cards (5-10 cards):
┌─────────────────────────────────┐
│ [Clinic Image] │ 渋谷動物病院   │
│                │ Shibuya Animal │
│                │ ★★★★☆ 4.2    │
│                │ 📍 0.8km       │
│                │ 🕐 営業中 Open │
│                │ [道案内 →]     │
└─────────────────────────────────┘

Bottom CTAs:
- 📹 "ビデオ診察 / Video Consultation" button (navy)
- 🚨 "緊急対応 / Emergency Now" button (red)

―――――――――――――――――――――――――――――――――――
PAGE 10: HEALTH REPORT SCREEN
―――――――――――――――――――――――――――――――――――
Time filter tabs: 1d | 1w | 1m | 3m | 6m | 4y

Report Sections (scrollable):
- Health Score Timeline (line chart)
- Temperature History (line chart, show 38.5°C average)
- Activity/Steps Bar Chart (weekly bars)
- Sleep Pattern Chart
- Vaccination Records list (checkboxes with dates)
- Last Vet Visit card

Two Action Cards side by side:
┌─────────────────┐  ┌─────────────────┐
│  📱 QR Report   │  │ 💰 Sell Report  │
│  獣医用QRコード  │  │ レポートを販売  │
│                 │  │ ① Vaccinations  │
│  [QR Code img]  │  │ ② Last checkup  │
│                 │  │ ③ Annual data   │
│ New QR each time│  │                 │
│ [生成 Generate] │  │ [PDF Export]    │
└─────────────────┘  └─────────────────┘

―――――――――――――――――――――――――――――――――――
PAGE 11: COMMUNITY SCREEN (Reddit-style)
―――――――――――――――――――――――――――――――――――
Header: "コミュニティ / Community" + ✏️ Create Post button

Sub-community filter chips (horizontal scroll):
すべて All | 柴犬部 | プードル部 | 
迷子情報 Lost Dogs | 獣医Q&A | 東京 | 大阪

Post Cards:
- User avatar + username + dog breed badge
- Time ago (Japanese: "3時間前")
- Post title (bilingual)
- Post preview text (2 lines)
- Optional image thumbnail
- 👍 [24] Upvote | 💬 [8] Comments | 🔗 Share
- Flair tag: #健康 or #しつけ or #グルーミング

Sample Posts to mockup:
1. "柴犬の体温が少し高いのですが / My Shiba's temp seems high"
   Flair: #健康 Health | 👍 47 | 💬 12
   
2. "東京でおすすめの獣医さん / Recommended vet in Tokyo?"
   Flair: #獣医Q&A | 👍 23 | 💬 34

3. "うちの子の毎日の散歩ルーティン / My dog's daily walk routine"
   Flair: #日常 Daily | 👍 89 | 💬 6

Individual Post View:
- Full post content
- Collar sensor data share card (special component)
- Nested comment thread
- "プロに聞く / Ask a Pro Vet" sticky CTA

―――――――――――――――――――――――――――――――――――
PAGE 12: AI CHATBOT SCREEN
―――――――――――――――――――――――――――――――――――
Header:
- AI mascot avatar (shiba inu in doctor coat, kawaii)
- "ワンケアAI / WanCare AI"
- Online status: green dot

Chat Window:
- User bubbles: right side, navy blue, white text
- AI bubbles: left side, white card, dark text
- AI responses can contain:
  → Inline health score card
  → Inline graph/chart
  → Navigation suggestion cards
  → Image analysis results

Quick Reply Chips (above input):
健康確認 Health Check | ワクチン Vaccine | 
近くの獣医 Find Vet | 緊急 Emergency

Sample Chat to Display:
User: "うちの犬の全体的な健康状態を教えて"
AI: [Health Summary Card]
    スコア: 87/100 ✓ 
    [Mini bar charts for each sensor]
    "全体的に健康です！/ Overall healthy!"
    [View Full Report button]

User: [Uploaded image of dog's skin]
AI: "画像を分析中... / Analyzing image..."
    "分析完了 / Analysis Complete"
    "皮膚の状態: 正常 / Skin: Normal ✓"

Input Bar (fixed bottom):
- 📷 Camera/Image upload button
- 🎤 Voice input button  
- Text input: "メッセージを入力 / Type a message"
- ➤ Send button (navy)

Elderly Mode Toggle (top right settings icon):
- Larger font
- Simpler language
- Profile navigation cards shown

―――――――――――――――――――――――――――――――――――
PAGE 13: BREED ENCYCLOPEDIA
―――――――――――――――――――――――――――――――――――
Search: "犬種を検索 / Search any breed"

Filter chips:
小型犬 Small | 中型犬 Medium | 大型犬 Large | 
ミックス Mixed | 🇯🇵 人気順 Japan Popular

Breed Grid (2 columns):
Each card:
- Breed illustration/avatar
- JP name (large)
- EN name (small, gray)
- Japan popularity rank badge: "🏅 #1 人気"
- Tap → Individual breed page

Individual Breed Page:
- Hero breed image/illustration
- Breed name JP + EN + Kanji
- Stat bars (0-100):
  エネルギー Energy ████████░░ 80%
  友好性 Friendliness █████████░ 90%
  訓練性 Trainability ███████░░░ 70%
  手入れ Grooming ██████░░░░ 60%
- Common health risks in Japan (list)
- "このコをコミュニティで見る / See in Community" button

―――――――――――――――――――――――――――――――――――
PAGE 14: SETTINGS SCREEN
―――――――――――――――――――――――――――――――――――
Profile Section:
- Circular avatar + edit icon
- "メールを変更 / Change Email"
- "名前を変更 / Change Name"
- "パスワード変更 / Change Password"

Appearance:
- 🌞 ライトモード / Light Mode — toggle
- 🌙 ダークモード / Dark Mode — toggle
(Live preview changes when toggled)

Language:
- Language selector (3 options, same as onboarding)

Notifications:
- 🚨 Emergency Alerts (locked ON, cannot disable)
- 📊 Daily Insights toggle
- 💬 Community Replies toggle
- 🏥 Vet Reminders toggle

Data & Export:
- "データをエクスポート / Export All Data" button
- Privacy settings link

Pro Features Card:
- 👑 Gold gradient card
- "プロプランにアップグレード / Upgrade to Pro"
- Feature list with checkmarks
- "月額 ¥980 / month" pricing

Danger Zone (bottom, separated):
- "アカウントを削除 / Delete Account" red text button

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 NAVIGATION & INTERACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Bottom navigation always visible (except onboarding)
- Active tab: sakura pink icon + underline
- Inactive tabs: gray
- Smooth page transitions
- Bottom sheets slide up with backdrop overlay
- SOS button always visible, pulsing red glow
- Dark mode toggle applies to entire app instantly
- All forms have proper validation messages in Japanese

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ TECH STACK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- React with TypeScript
- Tailwind CSS for styling
- Shadcn/ui for base components
- Recharts for all graphs and charts
- Lucide React for icons
- React Router for navigation
- Framer Motion for animations
- Use mock/dummy data for all sensor values
- Store language preference in localStorage
- Store dark mode preference in localStorage

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🇯🇵 JAPAN-SPECIFIC REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- All 47 Japan prefectures in dropdown
- LINE social login button (green, prominent)
- Japanese date format: YYYY年MM月DD日
- Metric units only (kg, °C, km)
- Currency: ¥ (Yen)
- Japanese popular dog breeds featured first
- Bilingual text throughout (JP primary, EN secondary)
- Dense information layout acceptable
- Elderly-friendly: large touch targets, clear fonts
- Sakura/paw print decorative elements throughout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📱 IMPORTANT NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Mobile-first design (375px - 430px width)
- All sensor data uses realistic mock values
- Charts use dummy data but look realistic
- QR code: show placeholder QR image
- Map: show styled placeholder map component
- Make it look like a REAL production app
- Every screen should feel complete, not wireframe-like
- Add subtle animations (fade in cards, pulse on live indicators)
- The overall feel should be: trustworthy, kawaii, medical-grade


## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
