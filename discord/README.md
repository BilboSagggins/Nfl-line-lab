# NFL Line Lab Discord bot

Posts this week's NFL games in a channel and edits them with live scores (edits never ping anyone).
Slash commands: `/games`, `/score team`, `/lines team`, `/setup`, `/stop`.

Runs free on a Cloudflare Worker. GitHub deploys it automatically whenever `discord/` changes.

## One-time setup

### 1. Create the bot in Discord
1. Go to **discord.com/developers/applications** → **New Application** → name it (e.g. *NFL Line Lab*).
2. **General Information**: copy **Application ID** and **Public Key**.
3. **Bot** → **Reset Token** → copy the token (shown once).
4. **Installation** (or OAuth2 → URL Generator): scopes `bot` + `applications.commands`; bot permissions **View Channels**, **Send Messages**, **Embed Links**, **Read Message History**. Open the link and add the bot to your server.

### 2. Create a free Cloudflare account
1. Sign up at **dash.cloudflare.com**, open **Workers & Pages** once (this sets up your free `workers.dev` address).
2. Copy your **Account ID** (right side of the Workers & Pages overview, or the URL after `dash.cloudflare.com/`).
3. **My Profile → API Tokens → Create Token → "Edit Cloudflare Workers"** template → Continue → Create. Copy the token.

### 3. Add five secrets to this GitHub repo
Settings → Secrets and variables → Actions → New repository secret:

| Name | Value |
|---|---|
| `DISCORD_APP_ID` | Application ID |
| `DISCORD_PUBLIC_KEY` | Public Key |
| `DISCORD_TOKEN` | Bot token |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare Account ID |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token |

### 4. Deploy
Actions → **Deploy Discord bot** → **Run workflow**. When it finishes, the run summary shows the bot's URL.

### 5. Connect Discord to the bot
Developer Portal → your app → **General Information** → **Interactions Endpoint URL** → paste the URL → Save.

### 6. Turn on the scoreboard
In your server, go to the channel you want and type **/setup**.
