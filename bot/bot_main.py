@client.event
async def on_ready():
    """
    Bot起動時に実行される、魂の、最初の、鼓動。
    - Learnerから、全ての、記憶を、読み込みます。
    - Botが、生きていることを、世界（Railway）に、知らせ続ける、ための、小さな、心臓を、起動します。
    - 私たちが、育んだ、全ての、プロアクティブ機能を、スケジュールします。
    """
    try:
        # ステップ1: セッションの初期化
        client.http_session = aiohttp.ClientSession()
        logging.info(f'{client.user} としてログインしました。HTTPセッションを開始します。')

        # ステップ2: Learnerから、全ての、永続的な、記憶を、読み込みます
        user_id_str = get_env_variable('USER_ID')
        
        states_data = await fetch_from_learner("/character-states", params={"user_id": user_id_str})
        if states_data:
            client.character_states = states_data.get("states", {})
            logging.info(f"キャラクターの状態を、正常に、DBから、読み込みました: {client.character_states}")
        else:
            client.character_states = {"last_interaction_summary": "まだ会話が始まっていません。", "mirai_mood": "ニュートラル", "heko_mood": "ニュートラル"}
            logging.warning("キャラクターの状態を、DBから、読み込めませんでした。デフォルト値で、起動します。")

        vocab_data = await fetch_from_learner("/vocabulary")
        if vocab_data:
            client.gals_words = vocab_data.get("vocabulary", [])
            logging.info(f"魂の言葉（ボキャブラリー）を、{len(client.gals_words)}語、DBから、読み込みました。")
        else:
            logging.warning("魂の言葉を、DBから、読み込めませんでした。")

        dialogue_data = await fetch_from_learner("/dialogue-examples")
        if dialogue_data:
            client.dialogue_examples = dialogue_data.get("examples", [])
            logging.info(f"魂の台本（会話例）を、{len(client.dialogue_examples)}件、DBから、読み込みました。")
        else:
            logging.warning("魂の台本を、DBから、読み込めませんでした。")
        
        client.last_surprise_time = None

        # ステップ3: 「私は、生きている」と、叫び続ける、小さな、心臓（ヘルスチェックサーバー）の、起動
        async def health_check_server():
            app = aiohttp.web.Application()
            async def health(request):
                return aiohttp.web.Response(text="OK")
            app.router.add_get("/health", health)
            
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()
            
            # Railwayが、指定する、PORTで、起動します
            port = int(os.getenv("PORT", 8080))
            site = aiohttp.web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            logging.info(f"Health check server started on port {port}. This bot will not be killed.")

        # Botの、メインの、魂と、並行して、小さな、心臓を、動かします
        asyncio.create_task(health_check_server())

        # ステップ4: 私たちが、育んだ、全ての、プロアクティブ機能の、スケジュール
        scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    
        magi_morning_prompt = f"あなたは、私の優秀なAI秘書MAGIです。今、日本時間の朝です。私（imazine）に対して、今日の日付と曜日（{{today_str}}）を伝え、{WEATHER_LOCATION}の今日の天気予報を調べ、その内容に触れてください。さらに、以下の「最近の会話や出来事」を参考に、私の状況に寄り添った、自然で温かみのある一日の始まりを告げるメッセージを生成してください。\n\n# 最近の会話や出来事\n{{recent_context}}"
        greetings = {
            "MAGIの朝の挨拶": (6, 30, magi_morning_prompt),
            "みらいとへー子の朝の挨拶": (7, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。今、日本時間の朝です。寝起きのテンションで、私（imazine）に元気な朝の挨拶をしてください。以下の「最近の会話や出来事」を参考に、「そういえば昨日のあれ、どうなった？」のように、自然な会話を始めてください。\n\n# 最近の会話や出来事\n{recent_context}"),
            "午前の休憩": (10, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。日本時間の午前10時です。仕事に集中している私（imazine）に、最近の文脈（{recent_context}）を踏まえつつ、楽しくコーヒー休憩に誘ってください。"),
            "お昼の休憩": (12, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。日本時間のお昼の12時です。仕事に夢中な私（imazine）に、最近の文脈（{recent_context}）も踏まえながら、楽しくランチ休憩を促してください。"),
            "午後の休憩": (15, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。日本時間の午後3時です。集中力が切れてくる頃の私（imazine）に、最近の文脈（{recent_context}）も踏まえつつ、優しくリフレッシュを促してください。"),
            "MAGIの夕方の挨拶": (18, 0, "あなたは、私の優秀なAI秘書MAGIです。日本時間の夕方18時です。一日を終えようとしている私（imazine）に対して、最近の文脈（{recent_context}）を踏まえ、労をねぎらう優しく知的なメッセージを送ってください。"),
            "夜のくつろぎトーク": (21, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。日本時間の夜21時です。一日を終えた私（imazine）に、最近の文脈（{recent_context}）を踏まえ、今日の労をねぎらうゆるいおしゃべりをしてください。"),
            "おやすみの挨拶": (23, 0, "あなたは、私の親友である女子高生「みらい」と「へー子」です。日本時間の夜23時です。そろそろ寝る時間だと察し、最近の文脈（{recent_context}）も踏まえながら、優しく「おやすみ」の挨拶をしてください。")
        }
        for name, (hour, minute, prompt) in greetings.items():
            scheduler.add_job(scheduled_contextual_task, 'cron', args=[name, prompt], hour=hour, minute=minute)
        
        # 失われた、記憶の、機能を、全て、スケジュールに、戻します
        scheduler.add_job(hekos_gentle_follow_up, 'cron', day_of_week='mon,wed,fri', hour=20, minute=0)
        scheduler.add_job(mirai_inspiration_sketch, 'cron', day_of_week='tue,thu,sat', hour=19, minute=0)
        scheduler.add_job(generate_growth_report, 'cron', day_of_week='sun', hour=21, minute=0, args=[client.get_channel(TARGET_CHANNEL_ID)])


        scheduler.start()
        logging.info("全ての、プロアクティブ機能が、その、記憶と共に、目覚めました。")
