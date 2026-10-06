use crate::{
    cli::UndoArgs, commands::util::run_history, error::Result, response::SceneInfo,
    store::HistoryDirection,
};

pub fn run(args: UndoArgs) -> Result<(Option<SceneInfo>, serde_json::Value)> {
    run_history(args, HistoryDirection::Undo)
}
