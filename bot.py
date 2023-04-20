import asyncio
import os
import tempfile
import discord
import whisper

ATTACHMENT_DIR = tempfile.mkdtemp()

DEFAULT_MODEL = "small"
MODELS = {
    model_name: whisper.load_model(model_name)
    for model_name in ["small", "medium", "large"]
}
LANGUAGES = whisper.tokenizer.LANGUAGES
DEVICE = MODELS[DEFAULT_MODEL].device

assert all(model.device == DEVICE for model in MODELS.values())


async def attachment_to_text(attachment, model_name=None, language=None):
    model = MODELS[model_name or DEFAULT_MODEL]
    filename = os.path.join(ATTACHMENT_DIR, str(attachment.id))
    await attachment.save(filename)
    result = model.transcribe(filename, language=language, fp16=False)
    os.remove(filename)
    return result["text"]


async def attachment_to_langs(attachment, model_name=None):
    model = MODELS[model_name or DEFAULT_MODEL]
    filename = os.path.join(ATTACHMENT_DIR, str(attachment.id))
    await attachment.save(filename)
    audio = whisper.pad_or_trim(whisper.load_audio(filename))
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    os.remove(filename)
    _, probs = model.detect_language(mel)
    most_likely = sorted(probs, key=probs.get, reverse=True)[:25]
    return most_likely


class ModelLanguageSelect(discord.ui.Select):
    def __init__(self, langs):
        options = [
            discord.SelectOption(label=LANGUAGES[lang], value=lang, default=i == 0)
            for i, lang in enumerate(langs)
        ]
        super().__init__(options=options)


class ModelSizeSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label=label, default=label == DEFAULT_MODEL)
            for label in MODELS
        ]
        super().__init__(options=options)


class ModelSelectorView(discord.ui.View):
    def __init__(self, langs):
        super().__init__()
        self.language_selected = langs[0]
        model_language_select = ModelLanguageSelect(langs)
        model_language_select.callback = self.on_language_selected
        self.add_item(model_language_select)

        self.model_size = DEFAULT_MODEL
        model_size_select = ModelSizeSelect()
        model_size_select.callback = self.on_model_size_selected
        self.add_item(model_size_select)

        model_retranscribe = discord.ui.Button(
            label="Retranscribe", style=discord.ButtonStyle.primary
        )
        model_retranscribe.callback = self.on_retranscribe
        model_retranscribe.disabled = True
        self.add_item(model_retranscribe)

    async def on_language_selected(self, interaction: discord.Interaction):
        self.language_selected = self.children[0].values[0]
        self.children[2].disabled = False
        await interaction.response.edit_message(view=self)

    async def on_model_size_selected(self, interaction: discord.Interaction):
        self.model_size = self.children[1].values[0]
        self.children[2].disabled = False
        await interaction.response.edit_message(view=self)

    async def on_retranscribe(self, interaction: discord.Interaction):
        self.children[0].disabled = True
        self.children[1].disabled = True
        self.children[2].disabled = True
        await interaction.response.edit_message(view=self)

        audio_message = (
            interaction.message.reference.cached_message
            or await bot.get_channel(
                interaction.message.reference.channel_id
            ).fetch_message(interaction.message.reference.message_id)
        )
        assert len(audio_message.attachments) == 1
        self.children[0].disabled = False
        self.children[1].disabled = False
        await interaction.followup.edit_message(
            interaction.message.id,
            content=await attachment_to_text(
                audio_message.attachments[0],
                model_name=self.model_size,
                language=self.language_selected,
            ),
            view=self,
        )


async def add_model_selector(message: discord.Message, langs):
    view = ModelSelectorView(langs)
    await message.edit(view=view)


class Voice2Text(discord.Client):
    def __init__(self):
        intents = discord.Intents.none()
        # https://discordpy.readthedocs.io/en/stable/api.html?highlight=intents#discord.Intents.guilds
        # "It is highly advisable to leave this intent enabled for your bot to function."
        intents.guilds = True
        # do I need this if I already have message_content?
        intents.messages = True
        intents.message_content = True
        intents.reactions = True
        super().__init__(intents=intents)
        self.tree = discord.app_commands.CommandTree(self)

    async def on_message(self, message):
        if message.author == self.user:
            return

        # flag for voice message is (1 << 13)
        # it is not yet documented in
        # https://discord.com/developers/docs/resources/channel#message-object-message-flags
        if message.flags.value & (1 << 13):
            reply = await message.reply("_transcribing audio_")
            await reply.add_reaction("🗑️")
            assert len(message.attachments) == 1
            text = await attachment_to_text(message.attachments[0])
            await reply.edit(content=text)
            await reply.add_reaction("🚩")
            await asyncio.sleep(60)
            await asyncio.gather(
                reply.remove_reaction("🚩", self.user),
                reply.remove_reaction("🗑️", self.user),
            )
        elif (
            isinstance(message.channel, discord.DMChannel)
            and len(message.attachments) > 0
        ):
            for attachment in message.attachments:
                await message.reply(await attachment_to_text(attachment))

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.user.id:
            return

        message = await self.get_channel(payload.channel_id).fetch_message(
            payload.message_id
        )

        # Ignore messages not written by the bot
        if message.author != self.user:
            return

        if payload.emoji.name == "🗑️":
            await message.delete()
        elif payload.emoji.name == "🚩":
            # TODO: this seems to get resolved but I'm not sure of the
            # conditions under which a message gets resolved or not
            voice_message = message.reference.resolved
            assert len(voice_message.attachments) == 1
            langs = await attachment_to_langs(voice_message.attachments[0])
            await add_model_selector(message, langs)

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        if payload.user_id == self.user.id:
            return

        if payload.emoji.name == "🚩":
            message = await self.get_channel(payload.channel_id).fetch_message(
                payload.message_id
            )
            if message.author != self.user:
                return

            await message.edit(view=None)

    async def setup_hook(self):
        await self.tree.sync()


bot = Voice2Text()


@bot.tree.context_menu(name="Voice to text")
async def voice_to_text(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(thinking=True)
    followup = False
    for attachment in message.attachments:
        await interaction.followup.send(await attachment_to_text(attachment))
        followup = True
    # TODO: stop the defer thinking if there is no followup


bot.run(os.environ["DISCORD_TOKEN"])
os.removedirs(ATTACHMENT_DIR)
