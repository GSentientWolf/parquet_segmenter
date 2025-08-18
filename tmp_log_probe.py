from parquet_segmenter import log

# Emit log lines at various levels; output should go through the configured sink
log.logger.debug('debug output (may be filtered)')
log.logger.info('info output')
log.logger.warning('warning output')
log.logger.error('error output')
log.logger.critical('critical output')
print('EXTRA STDOUT LINE')
