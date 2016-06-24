# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
from hubspot.items import HubspotItem
import re


class TrustSpider(CrawlSpider):
    RETRY_TIMES = 250
    name = 'trust'
    allowed_domains = ['trustradius.com']
    start_urls = ['https://www.trustradius.com/products/hubspot/reviews']

    rules = (
        Rule(LinkExtractor(allow=r'/products/hubspot/reviews\?f=\d+'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        i = HubspotItem()
        title_list = response.xpath('//a[@class="permalink"]/strong/text()').extract()
        # print title_list
        print 'number of titles', len(title_list)
        rating_list = response.xpath('//span[@class="rating-stars-text"]/span/text()').extract()[2:]
        date_list = response.xpath('//span[@class="rating-stars-text"]/text()').extract()[1:]
        # print 'dates:', date_list
        print 'dates count:', len(date_list)
        # print rating_list
        print 'number of ratings', len(rating_list)
        # usage_list = response.xpath('//*[matches(@id, "question-\d+-response-body")]/div/text()').extract()
        # print usage_list
        # print 'number of usages', len(usage_list)
        review_list = response.xpath('//div[@class="col-xs-12 serp-result"]')
        print 'number of reviews', len(review_list)
        # print review_list
        # print len(review_list)
        for rev in range(len(review_list)):
            soup = BeautifulSoup(review_list[rev].extract(), 'html.parser')

            text = soup.find_all('div', {'class': "col-md-12 question-response-container"})
            review = []
            for sec in text:
                r = ", ".join(item.strip() for item in sec.find_all(text=True))
                review.append(r)

            i['Review'] = " ".join(review).replace(u'\xa0', u' ')

            i['Title'] = title_list[rev].split('"')[1]
            i['Rating'] = int(re.findall('\d+', rating_list[rev])[0])
            i['Date'] = str(date_list[rev].split('|')[1].replace(u'\xa0', u'')).strip()

            print '----------------------------------------------------------------------------------------------------------------'
            yield i

        print '================================================================================================================'