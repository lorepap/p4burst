import asyncio
import os
import signal
import subprocess
import logging

class TcpServer:
    """
    Unified TCP server: per ogni connessione, legge un "REQ" e risponde con `reply_size` byte prima di chiudere.
    Supporta opzionale cattura pcap e logging.
    """
    def __init__(self, ip, port, exp_id, reply_size,
                 capture_pcap=True, log=True, backlog=1024):
        self.ip = ip
        self.port = port
        self.exp_id = exp_id
        self.reply_size = reply_size
        self.capture_pcap = capture_pcap
        self.log = log
        self.backlog = backlog
        self.tcpdump_process = None
        self.shutdown_event = None

    def start(self):
        """Entry point sincrono: avvia il server fino a terminazione."""
        asyncio.run(self._serve())

    async def _serve(self):
        """Coroutine che imposta il listener e gestisce shutdown."""
        self.shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        # Gestione segnali
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._on_shutdown)

        # Avvia pcap se richiesto
        if self.capture_pcap and self.exp_id:
            self._start_packet_capture()

        # Avvia il server TCP
        server = await asyncio.start_server(
            self._handle_client,
            host=self.ip,
            port=self.port,
            backlog=self.backlog
        )
        if self.log:
            logging.info(f"[{self.ip}]: Listening on port {self.port} (reply_size={self.reply_size} B)")

        # Attendi shutdown
        await self.shutdown_event.wait()
        server.close()
        await server.wait_closed()

        # Ferma pcap
        if self.capture_pcap:
            self._stop_packet_capture()
        if self.log:
            logging.info(f"[{self.ip}]: Server on port {self.port} shutdown")

    def _on_shutdown(self):
        """Callback per segnali di terminazione."""
        if self.log:
            logging.info(f"[{self.ip}]: Shutdown signal received")
        self.shutdown_event.set()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Gestisce singola connessione: read REQ, send reply_size byte."""
        peer = writer.get_extra_info('peername')
        try:
            # Leggi richiesta (es. "REQ")
            await reader.read(1024)
            # Invia reply_size byte
            writer.write(b'\0' * self.reply_size)
            await writer.drain()
            if self.log:
                logging.debug(f"[{self.ip}]: Sent {self.reply_size} B to {peer}")
        except Exception as e:
            if self.log:
                logging.error(f"[{self.ip}]: Error handling client {peer}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    def _start_packet_capture(self):
        """Avvia tcpdump su qualsiasi interfaccia per la porta del server."""
        pcap_file = f"tmp/{self.exp_id}/unified_server_{self.ip}_{self.port}.pcap"
        os.makedirs(os.path.dirname(pcap_file), exist_ok=True)
        cmd = [
            "sudo", "-n", "tcpdump", "-i", "any",
            "-w", pcap_file,
            f"port {self.port}", "-s", "0", "-U", "-W", "1", "-C", "1000"
        ]
        try:
            if self.log:
                logging.info(f"[{self.ip}]: Starting pcap to {pcap_file}")
            self.tcpdump_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp
            )
        except Exception as e:
            if self.log:
                logging.error(f"[{self.ip}]: Failed to start pcap: {e}")

    def _stop_packet_capture(self):
        """Termina il processo tcpdump."""
        if not self.tcpdump_process:
            return
        if self.log:
            logging.info(f"[{self.ip}]: Stopping pcap")
        self.tcpdump_process.terminate()
        try:
            self.tcpdump_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.tcpdump_process.kill()
            self.tcpdump_process.wait(timeout=1)
        self.tcpdump_process = None


