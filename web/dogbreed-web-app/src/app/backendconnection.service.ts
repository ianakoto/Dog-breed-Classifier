import { Injectable } from '@angular/core';
import {HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import {API_URL} from '../app/env';
import { Observable } from 'rxjs';




@Injectable({
  providedIn: 'root'
})
export class BackendconnectionService {

  constructor(private http: HttpClient) { }



  private static _handleError(err: HttpErrorResponse | any) {
    return Observable.throw(err.message || 'Error: Unable to complete request.');
  }


    // post prediction

    postPredictor(mdata) {
      console.log(mdata);
      return this.perform('post', `/api/v1/predict`,  mdata);

    }


    perform(method, resource, data) {
      const formData: FormData = new FormData();
      formData.append('imageurl', data);
      const url = `${API_URL}${resource}`;

      const httpOptions = {
        headers: new HttpHeaders({
          'Content-Type':  'multipart/form-data'
        }),
        responseType: 'json'
      };

      switch (method) {
        case 'post':
          return this.http.post(url, formData);
        case 'get':
          return this.http.get(url, {responseType: 'text'});
        default:
          return this.http[method](url, data);
      }


    }


}
