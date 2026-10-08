# HTTPS

## Purpose

A phone or tablet allows the microphone only on a page with HTTPS (or on `localhost`).
A page at `http://192.168.1.20:8080` cannot use the microphone.
This guide shows how to make the application serve HTTPS on the local network.

The application stays on HTTP when you set no certificate. This is the default.

## The quick way: ./start

`./start` does all the steps of this guide on its own. It makes the local authority and the certificate (for every address of the machine, and for `hostname.local`),
renews the certificate when the address changes or when it ends in less than 30 days, sets the variables `SAARTHI_TLS_CERT` and `SAARTHI_TLS_KEY`, and starts the application with HTTPS.
It also serves the public certificate of the authority at `http://<address>:8081/ca.pem` for the phone, as long as it runs. Nothing else of `certs/` is served.
The steps below are for a manual setup.

## Settings

In `backend/config.toml`, section `[server]`:

| Key | Meaning |
|-----|---------|
| `tls_certfile` | The path of the certificate file (PEM). A relative path starts at the project folder |
| `tls_keyfile` | The path of the key file (PEM) |

Set both keys, or set none. The variables `SAARTHI_TLS_CERT` and `SAARTHI_TLS_KEY` replace the keys, and `./start` sets them. One key alone is an error. A path that is not a file stops the start with a clear message.
With both keys set, the log says `Serving on https://...`.

The folder `certs/` and the files `*.pem` and `*.key` are in `.gitignore`. Never commit a key.

## Make a certificate

The phone must trust the certificate. The clean way is a small local authority (CA).
You make it one time and install it on each phone. Then you make a certificate for the address of the device.

Choose one fixed address for the device: a fixed IP address on the router, or a name such as `saarthi.local`.
The certificate names this address. If the address changes, make a new certificate.
Replace `192.168.1.20` below with your address. For a name, use `DNS:saarthi.local` in place of `IP:192.168.1.20`.

```
mkdir -p certs && cd certs

# 1. The local authority. Keep ca.key private.
openssl req -x509 -newkey rsa:4096 -nodes -days 3650 \
  -keyout ca.key -out ca.pem -subj "/CN=Saarthi local CA"

# 2. The key and the request of the device.
openssl req -newkey rsa:2048 -nodes \
  -keyout saarthi.key -out saarthi.csr -subj "/CN=saarthi"

# 3. The certificate of the device. It is valid for 397 days.
printf "subjectAltName=IP:192.168.1.20\nextendedKeyUsage=serverAuth\n" > san.ext
openssl x509 -req -in saarthi.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
  -days 397 -sha256 -extfile san.ext -out saarthi.pem
```

Then set in `backend/config.toml`:

```
tls_certfile = "certs/saarthi.pem"
tls_keyfile = "certs/saarthi.key"
```

Start the application. Open `https://192.168.1.20:8080/find` in a browser.

A phone accepts a certificate for at most 397 days. Make a new one (steps 2 and 3) before it ends.

## Install the authority on a phone

Send `certs/ca.pem` (not the key files) to the phone. Use a USB cable or a local transfer. Do not use a cloud service.

- **Android:** Settings, Security, Install a certificate, CA certificate. Choose `ca.pem`.
  The menu names differ between makers.
- **iPhone or iPad:** Open the file and install the profile (Settings, Profile Downloaded).
  Then open Settings, General, About, Certificate Trust Settings, and turn on full trust for "Saarthi local CA".

Then open `https://<address>:8080/find` on the phone. The page must show no warning.

## Test the microphone

Open the browser console on the phone page, or use a test page. The check `window.isSecureContext` must be `true`.
The microphone itself comes in a later phase (stage Q0).

## Risks

- **ca.key.** Anyone who has it can make certificates that the phones trust. Keep it off the device, or delete it after step 3.
  The device needs only `saarthi.pem` and `saarthi.key`.
- **A changed address.** The browser shows a warning. Make a new certificate.
- **A clock that is wrong.** A device with a wrong date can make the certificate look invalid.
- **Not tested on a real phone.** The steps above are the standard way. Test them on the real phone and browser before you rely on them.
