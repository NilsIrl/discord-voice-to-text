import os
import discord
import whisper
from pathlib import Path

ATTACHMENT_DIR = Path("attachments")

DEFAULT_MODEL = "small"
MODELS = {
        "small": whisper.load_model("small"),
        }

def get_text(filename):
    result = MODELS[DEFAULT_MODEL].transcribe(str(filename))
    return result["text"]

class Voice2Text(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = discord.app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

bot = Voice2Text(intents=discord.Intents.default())

@bot.tree.context_menu(name="Voice to text")
async def voice_to_text(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(thinking=True)
    for attachment in message.attachments:
        filename = ATTACHMENT_DIR / str(attachment.id)
        await attachment.save(filename)
        await interaction.followup.send(get_text(filename))
        os.remove(filename)

bot.run(os.environ['DISCORD_TOKEN'])
