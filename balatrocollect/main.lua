--- STEAMODDED HEADER
--- MOD_NAME: datacollect
--- MOD_ID: datacollect-v0.1
--- MOD_AUTHOR: [Aidan-Jared]
--- MOD_DESCRIPTION: A datacollection API for Balatro
-- Code taken from besteon/balatrobot and modified

function SMODS.INIT.BALATROBOT()
	mw = SMODS.findModByID("datacollect-v0.1")

	-- Load the mod configuration
	assert(load(NFS.read(mw.path .. "config.lua")))()
	if not BALATRO_BOT_CONFIG.enabled then
		return
	end

	-- External libraries
	assert(load(NFS.read(mw.path .. "lib/list.lua")))()
	assert(load(NFS.read(mw.path .. "lib/hook.lua")))()
	assert(load(NFS.read(mw.path .. "lib/bitser.lua")))()
	assert(load(NFS.read(mw.path .. "lib/sock.lua")))()
	assert(load(NFS.read(mw.path .. "lib/json.lua")))()

	-- Mod specific files
	assert(load(NFS.read(mw.path .. "src/utils.lua")))()

	if BALATRO_BOT_CONFIG.passive_mode then
		-- only use passive api
		assert(load(NFS.read(mw.path .. "src/action_tracker.lua")))
		assert(load(NFS.read(mw.path .. "src/api_passive.lua")))
	else
		-- load full bot if not in passive
		assert(load(NFS.read(mw.path .. "src/bot.lua")))()
		assert(load(NFS.read(mw.path .. "src/middleware.lua")))()
		assert(load(NFS.read(mw.path .. "src/botlogger.lua")))()
		assert(load(NFS.read(mw.path .. "src/api.lua")))()
	end

	sendDebugMessage("datacollect-v0.1 loaded")

	if BALATRO_BOT_CONFIG.passive_mode then
		BalatrobotAPI.init()
	else

		Middleware.hookbalatro()

		Botlogger.path = mw.path
		Botlogger.init()
		BalatrobotAPI.init()
	end
end
