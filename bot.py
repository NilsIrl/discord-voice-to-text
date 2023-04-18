import os
import tempfile
import discord
import whisper
from pathlib import Path

ATTACHMENT_DIR = Path(tempfile.mkdtemp())

DEFAULT_MODEL = "small"
MODELS = {
    "small": whisper.load_model("small"),
}


def get_text(filename):
    result = MODELS[DEFAULT_MODEL].transcribe(str(filename), fp16=False)
    return result["text"]


async def attachment_to_text(attachment: discord.Attachment):
    filename = ATTACHMENT_DIR / str(attachment.id)
    await attachment.save(filename)
    text = get_text(filename)
    os.remove(filename)
    return text


class Voice2Text(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = discord.app_commands.CommandTree(self)

    async def on_message(self, message):
        if message.author == self.user:
            return

        # flag for voice message is (1 << 13)
        # it is not yet documented in https://discord.com/developers/docs/resources/channel#message-object-message-flags
        if (
            message.flags.value & (1 << 13)
            or isinstance(message.channel, discord.DMChannel)
            and len(message.attachments) > 0
        ):
            assert len(message.attachments) == 1
            await message.reply(await attachment_to_text(message.attachments[0]))

    async def setup_hook(self):
        await self.tree.sync()


intents = discord.Intents.default()
intents.message_content = True
bot = Voice2Text(intents=intents)


@bot.tree.context_menu(name="Voice to text")
async def voice_to_text(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(thinking=True)
    for attachment in message.attachments:
        await interaction.followup.send(await attachment_to_text(attachment))


bot.run(os.environ["DISCORD_TOKEN"])
os.removedirs(ATTACHMENT_DIR)
