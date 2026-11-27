# Security policy

## Reporting a vulnerability

Please don't open a public issue. Report the vulnerability privately: on this repo's **Security**
tab, click **Report a vulnerability**.

Say what you found, how to reproduce it, and which TracePress version you used. Remove API keys
and private traces from anything you send.

We'll reply as soon as we can and work with you on a fix. Please don't disclose the problem
publicly until the fix is released.

## By design

`tracepress serve` runs the pipelines that clients register, and a registration ships the
pipeline's code pickled, so anyone who can reach the server can run code on that machine. The server listens
on localhost by default, needs `--token` for any other `--host`, and refuses requests from web
pages. A way past any of these three protections is a vulnerability.
