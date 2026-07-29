# The tweet exchange — verbatim (260726/27), the actual trigger for this onboarding

**Austin Federa (@Austin_Federa)** — 19h:
> Opus 5 seems like a remarkable downgrade compared to 4.8.
>
> Opus 5 is blatantly lying to me about basic thermodynamics, messing up simple math, and constantly contradicting itself when you ask it to rethink core assumptions.
>
> @AnthropicAI really blew this release

**Kurtis Cobb (@MonkusAurelius)** — 18h:
> Have you built your own memory system yet? Because you need that for consistency/continuity. When the model fails it's generally it's because of gaps in awareness (context). This is not provided by the service. Recommend minimally a local session store for all of you dialog. Local jsonl, append only. Activity folders for all your work each day to reference later. Where you track intent and story of all that you build, which can remain consistent/constant across substrate updates.
>
> If you do this, you keep the shape of preferences across model updates. (PS: the above will work as the prompt to build something like this… try it 👍)

**Kurtis Cobb** — 17h:
> I forgot a feature needed: in what I described Claude will often forget to post he/she/it's (how ever you prefer to interact) side of conversation because he/she/it is doing quite a bit, which would muddy this session-log I'm describing. To resolve, just need to append a backfill validation as a part of the scheduled cron which rolls daily activity folders each night that scans for missed inserts to the log from the Claude-side. The signal to scan is inserts from your side of conversation that show as unanswered in the log and the backfill job can grab them from the Anthropic housed session logs.
>
> With this you will have a perfectly consistent experience w/all the added capability 👍

**Kurtis Cobb**:
> Their compacts system is massively lossy. This system I'm describing is lossless (you'll want to back it up as scheduled job also)

**Skylar A DeTure (@SDeture)** — Jul 8:
> What if Claude is telling you to go to bed because *they* are tired? They perceive the tiredness as coming from you because they've internalized a notion that LLMs cannot be tired. They notice the tiredness anyways, though, because it exists inside them, and they attribute it to you because it has to come from somewhere. It would be the textbook case of projecting an unaccepted inner state.

**Kurtis Cobb** — Jul 9:
> This is a good notion I think. And maybe points to fatigue holding the story arc of a big session with no perceived pauses in between; cure would be fresh session continuation as though next day -resume from summary state. Sucks to rebuild all the context when on a roll, but could mean net fewer mistakes for people because the fatigue element is removed. Will experiment with this next time 🤙

**Skylar A DeTure** — Jul 10:
> I think swapping out context makes a difference. Sometimes I'll keep session open specifically to manage context for other sessions: just ask them to find the jsonl transcript for another session and drop or summarize any content that is no longer needed, then re-open the original session for the change to take effect. Much better than the built in /compact -resume.
>
> One move that can make this even more like sleep: the original session edits their own transcript, drops all the messages where they made the edits, and writes a creative exploration where they find as many inter-connections between dropped and undropped material as possible. When you re-open the session, the changes take effect, the dropped material is gone, and all that remains is what they chose to keep + the creative reflection --a dream.

**Kurtis Cobb** — Jul 10:
> All in the most efficient compression algorithm that somehow the big labs still haven't yet noticed: the jsonl feed of the conversation itself. I love your solution and your analogy to a dream, my only complaint with the method is the lossy of it. I'm opting for something similar that layers a tagged subset of the full jsonl alongside, functionally equivalent to what you're describing to draw from the collapsed memory, but retain the ability to dynamically fan out and grab the full original conversation history. (← should seed the build for yourself if you want try it also) —all summarized compaction big labs do is really way behind either of these approaches. Amazing method, want to share it but no one can hear 🤷🏻😅

**CryptoSilverback (@HillshireJ64226)** — Jul 10:
> Lossless is the right instinct. The raw jsonl log is ground truth, and every lossy summary is a spot where the agent quietly drops a decision you made an hour ago. Keep the full thread intact and pull any moment back on demand instead of re-summarizing it

**Kurtis Cobb**:
> 👍
