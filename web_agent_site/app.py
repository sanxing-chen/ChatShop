import argparse, json, logging, random
from pathlib import Path
import json
import sqlite3
from ast import literal_eval

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    jsonify,
    g,
)
from flask_cors import CORS

from rich import print

from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    convert_web_app_string_to_var,
    get_top_n_product_from_keywords,
    get_product_per_page,
    map_action_to_html,
    END_BUTTON,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (
    generate_mturk_code,
    setup_logger,
    DEFAULT_FILE_PATH,
    DEBUG_PROD_SIZE,
)

import gymnasium as gym

from web_agent_site.envs import IPOTextEnv
from web_agent_site.models import Agent

app = Flask(__name__)
CORS(app)

search_engine = None
all_products = None
product_item_dict = None
product_prices = None
attribute_to_asins = None
goals = None
weights = None

user_sessions = dict()
user_log_dir = None

DATABASE = "annotation.db"


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        init_db(db)
        # configure conn to return dict
        db.row_factory = sqlite3.Row
    return db


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def home():
    return "welcome to chatshop!"


if search_engine is None:
    all_products, product_item_dict, product_prices, attribute_to_asins = load_products(
        filepath=DEFAULT_FILE_PATH, num_products=DEBUG_PROD_SIZE
    )
    search_engine = init_search_engine(num_products=DEBUG_PROD_SIZE)
    goals = json.load(open("data/chatshop_goals.json"))
    weights = [goal["weight"] for goal in goals]


def init_db(db):
    c = db.cursor()

    c.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                session_id TEXT UNIQUE,
                current_gid INTEGER,
                num_products INTEGER,
                num_question INTEGER,
                game_mode   INTEGER,
                model_version TEXT,
                strategy TEXT,
                qmode TEXT,
                cot BOOLEAN
    )"""
    )

    c.execute(
        """CREATE TABLE IF NOT EXISTS annotation (
              session_id TEXT,
              qid INTEGER,
              review BOOLEAN,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
              """
    )

    # insert one session
    # c.execute('INSERT OR IGNORE INTO sessions (session_id, current_gid, num_products, num_question, game_mode, model_version, strategy, qmode, cot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', ('bbb02d8fcc3746b2b969deb9658e5552', 666, 20, 5, 2, 'gpt-3.5-turbo-1106', 'interleave', 'open', False))

    db.commit()


@app.route("/api/search_results", methods=["GET"])
def api_search_results():
    if "keywords" not in request.args or "gid" not in request.args:
        return jsonify([])
    keywords = request.args.get("keywords").lower().split(" ")
    top_n_products = get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins,
    )

    goal = goals[int(request.args.get("gid"))]
    rewards = [
        get_reward(
            product,
            goal,
            price=product_prices[product["asin"]],
            # in ChatShop, we don't require the agent to select options,
            # reward is calculated based on the availability of options
            options=dict(enumerate(sum(product["options"].values(), []))),
            verbose=True,
        )
        for product in top_n_products
    ]
    rewards = [{"r_total": r[0], **r[1]} for r in rewards]

    combined = [
        {
            **product,
            **reward,
        }
        for product, reward in zip(top_n_products, rewards)
    ]

    response = jsonify(combined)
    return response


@app.route("/api/get_goals", methods=["GET"])
def api_get_goals():
    if "gid" not in request.args:
        return jsonify([])
    gid = int(request.args.get("gid"))
    data = goals[gid]
    data["details"] = product_item_dict[data["asin"]]
    response = jsonify(data)
    return response

# The following code is used for human annotation

@app.route("/api/get_history", methods=["GET"])
def api_get_history():
    if "id" not in request.args or "path" not in request.args:
        return jsonify({"status": "error", "message": "Missing id or path."})
    id = int(request.args.get("id"))
    path = request.args.get("path")

    path = Path(path)
    if not path.exists():
        return jsonify({"status": "error", "message": f"Path {path} does not exist."})

    with open(path, "r") as f:
        history = json.load(f)

    if id >= len(history):
        return jsonify({"status": "error", "message": f"Id {id} is out of bound."})

    response = jsonify({"status": "success", "data": history[id]})
    return response


def session_output(session_id):
    conf = query_db(
        "SELECT * FROM sessions WHERE session_id=?", (session_id,), one=True
    )
    d = user_sessions[session_id]
    return {**conf, **{k: v for k, v in d.items() if not k.startswith("env")}}


@app.route("/api/session", methods=["POST"])
def api_session():
    request_data = request.get_json()
    session_id = request_data.get("session_id", "")

    # get env and agent configures from database
    conf = query_db(
        "SELECT * FROM sessions WHERE session_id=?", (session_id,), one=True
    )
    if conf is None:
        return jsonify({"status": "error", "message": "Invalid session_id."})

    current_gid = conf["current_gid"]

    if (
        session_id not in user_sessions
        or "gid" in request_data
        or current_gid != user_sessions[session_id]["info"]["sample_id"]
    ):
        # init a new game
        if "gid" in request_data:
            gid = request_data.get("gid")
            db = get_db()
            c = db.cursor()
            c.execute(
                "UPDATE sessions SET current_gid=? WHERE session_id=?",
                (gid, session_id),
            )
            db.commit()
            current_gid = gid

        env = gym.make(
            "IPOEnv-v0",
            num_products=conf["num_products"],
            num_question=conf["num_question"],
            game_mode=conf["game_mode"],
        )

        obs, info = env.reset(gid=current_gid)
        agent = Agent(
            conf["model_version"],
            qmode=conf["qmode"],
            cot=conf["cot"],
            strategy=conf["strategy"],
        )

        action = agent.take_action(obs, info["question_budget"])
        while not (action.startswith("question") or action.startswith("opinion")):
            obs, reward, done, _, info = env.step(action)
            if done:
                break
            action = agent.take_action(obs, info["question_budget"])

        user_sessions[session_id] = {
            "env": env,
            "env_agent": agent,
            "messages": agent.messages,
            "done": done,
            "reviewed": False,
            "info": info,
        }

    response = jsonify({"status": "success", "data": session_output(session_id)})
    return response


@app.route("/api/submit_answer", methods=["POST"])
def api_submit_answer():
    request_data = request.get_json()
    if "session_id" not in request_data:
        return jsonify({"status": "error", "message": "Missing session_id."})
    if "answer" not in request_data:
        return jsonify({"status": "error", "message": "Missing answer."})
    session_id = request_data["session_id"]
    if session_id not in user_sessions:
        return jsonify({"status": "error", "message": "Invalid session_id."})

    env = user_sessions[session_id]["env"]
    agent = user_sessions[session_id]["env_agent"]
    done = user_sessions[session_id]["done"]
    if done:
        return jsonify(
            {"status": "error", "message": "Session is done, next game please."}
        )

    answer = request_data["answer"]
    action = agent.last_action()
    env.external_answer(answer)
    obs, reward, done, _, info = env.step(action)
    action = agent.take_action(obs, info["question_budget"])
    while not (action.startswith("question") or action.startswith("opinion")):
        obs, reward, done, _, info = env.step(action)
        if done:
            break
        action = agent.take_action(obs, info["question_budget"])

    user_sessions[session_id]["done"] = done
    user_sessions[session_id]["info"] = info

    response = jsonify({"status": "success", "data": session_output(session_id)})

    return response


@app.route("/api/submit_review", methods=["POST"])
def api_submit_review():
    request_data = request.get_json()
    if "session_id" not in request_data:
        return jsonify({"status": "error", "message": "Missing session_id."})
    if "review" not in request_data:
        return jsonify({"status": "error", "message": "Missing review."})
    session_id = request_data["session_id"]
    if session_id not in user_sessions:
        return jsonify({"status": "error", "message": "Invalid session_id."})
    if not user_sessions[session_id]["done"]:
        return jsonify({"status": "error", "message": "Session not done."})
    if user_sessions[session_id]["reviewed"]:
        return jsonify({"status": "error", "message": "Already reviewed."})

    review = request_data["review"]
    if isinstance(review, str):
        review = review.lower() in ["true", "1", "yes"]

    user_sessions[session_id]["reviewed"] = True

    agent = user_sessions[session_id]["env_agent"]
    info = user_sessions[session_id]["info"]

    # save to db
    db = get_db()
    c = db.cursor()
    c.execute(
        "INSERT INTO annotation (session_id, qid, review) VALUES (?, ?, ?)",
        (session_id, info["sample_id"], review),
    )
    db.commit()

    # count the number of reviews
    num_reviews = query_db(
        "SELECT COUNT(*) FROM annotation WHERE session_id=?", (session_id,), one=True
    )[0]

    # save to file
    save = {
        "agent": agent.model,
        "agent_messages": agent.messages,
        "info": info,
        "review": review,
    }
    with open(f"logs/{session_id}.jsonl", "a") as f:
        print("saving to", f.name)
        f.write(json.dumps(save) + "\n")

    response = jsonify({"status": "success", "data": {"num_reviews": num_reviews}})

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChatShop flask app backend configuration"
    )

    args = parser.parse_args()

    app.run(host="0.0.0.0", port=3000)
