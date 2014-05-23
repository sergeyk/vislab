import optparse
import json
import flask
import tornado.wsgi
import tornado.httpserver


def make_json_response(body, status_code=200):
    resp = flask.make_response(json.dumps(body))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


def get_query_args(necessary=None, defaults=None, types=None):
    """
    Parameters
    ----------
    necessary: list [None]
        These arg names must be present.
    defaults: dict [None]
        arg_name: default value
    types: dict [None]
        arg_name: type
    """
    args = dict((key, val)
                for key, val in flask.request.args.iteritems()
                if val not in [None, 'None', ''])
    print("[get_query_args] Request args: {}".format(args))

    if necessary is not None:
        if not all([(arg_name in args) for arg_name in args.keys()]):
            raise Exception("A necessary argument was not provided.")

    if defaults is not None:
        for key, val in defaults.items():
            if key not in args:
                args[key] = val
    if types is not None:
        for arg, type_ in types.items():
            if arg in args:
                args[arg] = type_(args[arg])
    return args


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    opts, args = parser.parse_args()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)
